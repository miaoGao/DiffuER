import math
import torch

import torch.nn.functional as F

from torch import nn
from transformers import BertModel, AutoConfig, AutoModel, BartForConditionalGeneration, AutoTokenizer, BartConfig

from model_utils.gaussian_diffusion import timestep_embedding
from model_utils.CrossAttention import BasicTransformerBlock

from transformers import logging
logging.set_verbosity_error()

import logging
logger = logging.getLogger(__name__)


class SimpleSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, embedding_dim, num_embeddings=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        emb[0, :] = 0
        self.weight = emb.cuda()

    def forward(self, positions):
        return self.weight.index_select(0, positions.contiguous().view(-1)).view(positions.size() + (self.embedding_dim,)).detach()


class CrossAttention_Diffusion_LM(nn.Module):
    def __init__(self, config, out_channels, vocab_size=None):
        super().__init__()
        self.config = config
        self.out_channels = out_channels
        self.time_channels = config.time_channels
        self.vocab_size = vocab_size
        
        model_cfg = AutoConfig.from_pretrained(config.model.name)
        
        if vocab_size is not None:
            model_cfg.vocab_size = vocab_size
            
        input_size = config.in_channels
        if config.self_condition:
            input_size = config.in_channels * 2

        model_cfg.intermediate_size = config.intermediate_size
        model_cfg.hidden_size = int(model_cfg.intermediate_size / 4)
        model_cfg.max_position_embeddings = config.max_pos_len
        model_cfg.num_hidden_layers = 6
        model_cfg.num_attention_heads = config.num_attention_heads
        model_cfg.hidden_dropout_prob = config.dropout
        model_cfg.attention_probs_dropout_prob = config.att_dropout
        
        self.hidden_size = model_cfg.hidden_size
        self.encoder = BertModel(config=model_cfg)

        if config.model.use_teacher:
            self.bert2diff = nn.Linear(768, config.in_channels)

        if (input_size != self.hidden_size) or (self.config.self_condition):
            # input transform layer
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_size, self.hidden_size * 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
        if out_channels != self.hidden_size:
            # output transform layer
            self.output_down_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size * 2, out_channels)
            )

        if config.pred_len:
            self.length_embedding = nn.Embedding(config.tgt_len + 1, self.hidden_size, None)
            self.word_embedding = nn.Embedding(model_cfg.vocab_size, config.in_channels, config.pad_value)
        else:
            self.word_embedding = nn.Embedding(model_cfg.vocab_size, config.in_channels, None)

        self.register_buffer("position_ids", torch.arange(model_cfg.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(model_cfg.max_position_embeddings, self.hidden_size)
        
        self.embed_scale = math.sqrt(config.in_channels) if self.config.scale_embedding else 1.0
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=model_cfg.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        if self.config.predict_x_start:
            self.lm_transform = nn.Sequential(
                nn.Linear(config.in_channels, config.in_channels),
                nn.GELU(),
                nn.LayerNorm(config.in_channels, eps=model_cfg.layer_norm_eps)
            )
        self.lm_head = nn.Linear(config.in_channels, model_cfg.vocab_size)
        # lm_head and word_embedding share parameters（important）
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        if not config.time_att:
            # time embedding layer
            self.time_trans = nn.Sequential(
                nn.Linear(self.time_channels, self.time_channels * 4),
                nn.SiLU(),
                nn.Linear(self.time_channels * 4, self.hidden_size),
            )

        # define cross attention transformer block(6 layer)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    config=config,
                    dim=self.hidden_size,
                    num_attention_heads=model_cfg.num_attention_heads,
                    attention_head_dim=self.hidden_size // model_cfg.num_attention_heads,
                    dropout=config.dropout,
                    attention_dropout=model_cfg.attention_probs_dropout_prob,
                    cross_attention_dim=self.hidden_size,
                    activation_fn="geglu",
                    attention_bias=False,
                )
                for d in range(model_cfg.num_hidden_layers)
            ]
        )

        # def aux ar model
        if config.model.aux_ar_model:
            aux_config = BartConfig()
            aux_config.vocab_size = vocab_size
            aux_config.encoder_layers = config.model.aux_ar_model_layers
            aux_config.encoder_ffn_dim = config.intermediate_size
            aux_config.encoder_attention_heads = config.num_attention_heads
            aux_config.decoder_layers = config.model.aux_ar_model_layers
            aux_config.decoder_ffn_dim = config.intermediate_size
            aux_config.decoder_attention_heads = config.num_attention_heads
            aux_config.d_model = config.in_channels
            self.aux_ar_model = BartForConditionalGeneration(aux_config)
            self.aux_ar_model.model.shared = self.word_embedding
            
        logger.info(model_cfg)

    def encode(self, sentences, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #  tokenizer.encode_plus  batch_encode_plus  input_ids
        encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        #  encoding['input_ids']
        input_ids_tensor = encoding['input_ids']
        print("=============embed scale==",self.word_embedding(input_ids_tensor).shape)
        return torch.mean(self.word_embedding(input_ids_tensor), dim=1)

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids) * self.embed_scale

    def get_logits(self, hidden_repr):
        if self.config.predict_x_start:
            hidden_repr = self.lm_transform(hidden_repr)
        return self.lm_head(hidden_repr)
        
    def get_pred_len(self, encoder_hidden_states, src_masks=None, normalize=False):
        """
        mean pooling / get the representation of <length>
        encoder_hidden_states: [bs, seq_len, hz]
        src_masks: [bs, seq_len]
        """
        if self.config.pred_len_strategy == 'token_embed':
            enc_feats = encoder_hidden_states[:, 0, :]  #  [bs, hz]
        elif self.config.pred_len_strategy == 'mean_pool':
            # [bs, seq_len, hz] / [bs, 1, 1] * [bs, seq_len, 1]
            enc_feats = (
                (encoder_hidden_states / src_masks.sum(1)[:, None, None]) * src_masks[:, :, None]
            ).sum(1)
        length_out = F.linear(enc_feats, self.length_embedding.weight)

        return F.log_softmax(length_out, -1) if normalize else length_out

    def forward(self, tgt_emb, timesteps, src_attention_mask, tgt_attention_mask, 
                src_input_ids=None, encoder_hidden_states=None, tgt_length=None, x_self_cond=None,
                src_head_mask=None,
            ):
        # 1. prepare encoder output
        length_out = None
        if encoder_hidden_states is None:
            # Only for Training
            out = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask, head_mask=src_head_mask)
            # In order to facilitate gradient return.
            encoder_hidden_states = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)
            
            if self.config.pred_len:
                length_out = self.get_pred_len(encoder_hidden_states=encoder_hidden_states,
                                               src_masks=src_attention_mask)

        # 2. prepare decoder input embedding
        if self.config.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(tgt_emb)
            tgt_emb = torch.cat((x_self_cond, tgt_emb), dim=-1)

        if (tgt_emb.size(-1) != self.hidden_size) or (self.config.self_condition):
            tgt_emb = self.input_up_proj(tgt_emb)

        # 3. prepare time & position embedding
        # xy: [bs, seq_len, hz] / baseline: [bs, hz]
        time_emb = timestep_embedding(self.config, timesteps, self.time_channels)
        if self.config.schedule_sampler == 'uniform':
                time_emb = time_emb.unsqueeze(1).expand(-1, tgt_emb.size(1), -1)

        position_ids = self.position_ids[:, : tgt_emb.size(1)]
        pos_emb = self.position_embeddings(position_ids)

        # 4. Dropout & LayerNorm
        if tgt_length is not None:
            len_emb = self.length_embedding(tgt_length).unsqueeze(1).expand(-1, tgt_emb.size(1), -1)
            decoder_input = len_emb + pos_emb + tgt_emb 
        else:
            decoder_input = pos_emb + tgt_emb 
        if not self.config.time_att:
            time_emb = self.time_trans(time_emb)
            decoder_input = decoder_input + time_emb
        hidden_states = self.dropout(self.LayerNorm(decoder_input))

        # 5. Decoder Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                context=encoder_hidden_states,
                time_emb=time_emb if self.config.time_att else None,
                encoder_key_padding_mask=src_attention_mask,
                tgt_padding_mask=tgt_attention_mask,
            )

        # 6. Output Projection
        if hidden_states.size(-1) != self.out_channels:
            hidden_states = self.output_down_proj(hidden_states)

        output = hidden_states.type(tgt_emb.dtype)  
        return output, length_out
