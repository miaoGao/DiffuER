# when generate token, consider attention mask with syntax infomation


import os
import hydra
import torch
import logging
import random
import pickle
from hydra import initialize, compose

import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from utils import load_states_from_checkpoint
from data_utils.s2s_dataset import load_jsonl_data, S2S_dataset
from data_utils.tokenizer_utils import create_tokenizer
from model_utils.create_model import create_model, create_gaussian_diffusion
import jsonlines

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ：text_emb
def denoised_fn_round(config, emb_model, text_emb, t):
    down_proj_emb = emb_model.weight  # (vocab_size, embed_dim)

    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb):
        emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # (vocab, 1)
        text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # (emb_dim, bs*seqlen)
        arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # (bs*seqlen, 1)
        # down_proj_emb: (vocab, emb_dim), text_emb_t:(emb_dim, bs*seqlen)
        # a+b automatically broadcasts to the same dimension i.e. (vocab, bs*seqlen)
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t) 
        dist = torch.clamp(dist, 0.0, np.inf)  # Limit the value of input to [min, max].
        # Select the smallest distance in the vocab dimension, 
        # that is, select bs*seq_len most likely words from all vocabs.
        topk_out = torch.topk(-dist, k=1, dim=0)

        return topk_out.values, topk_out.indices  # logits, token_id (1, bs*seq_len)

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device))
    rounded_tokens = indices[0]  # (bs*seq_len,)
    new_embeds = emb_model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds


# ：（？）
def split_data(data, log=False):
    shard_size = len(data) // 1
    start_idx = shard_size
    end_idx = start_idx + shard_size
    end_idx = len(data)
    data_piece = data[start_idx: end_idx]

    print(" shard_size ==",shard_size)
    print(" start_idx ==",start_idx)    
    print(" end_idx ==",end_idx)
    
    if log:
        logger.info(f'generation for {len(data_piece)} text from idx {start_idx} to {end_idx}')
    
    return data

def bpe_batch_decode(sample_id_tensor, tokenizer):
    result = []
    for sample_id in sample_id_tensor:
        text = tokenizer.decode(sample_id.tolist(), skip_special_tokens=True)
        for token in ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]:
            text = text.replace(token, '')
        result.append(text)
    return result

# ：
@hydra.main(version_base=None, config_path="../", config_name="config")
def main(config):

    # （）
    # local_rank = int(os.environ["LOCAL_RANK"])
    
    # 

    config.exp.dir = os.path.join(config.exp.root, config.data.name, config.exp.name)
    generate_path = os.path.join(config.exp.dir, str(config.load_step))
    if config.load_from_ema:
        generate_path += ('_ema_' + str(config.ema_rate))
    if config.clip_denoised:
        generate_path += '_clip_denoised_'
    if config.infer_self_condition:
        generate_path += '_selfcond_'
    if config.skip_sample:
        generate_path += '_skip_'
    if config.ddim_sample:
        generate_path += '_ddim_'

    if config.schedule_sampler == 'xy_uniform':
        generate_path += ('_xy_' + str(config.gen_timesteps))
    else:
        generate_path += ('_un_' + str(config.skip_timestep))

    if (not os.path.exists(generate_path)):
        os.makedirs(generate_path)

    # CUDA，，GPU，CPU。

    # （）
    # torch.cuda.set_device(local_rank)  # ddp setting
    # dist.init_process_group(backend="nccl")  # ddp setting

    # 

    set_seed(config.exp.seed + int(0))  # seed setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.visible_device)

    # tokenizer
    
    if config.data.name in ['iwslt14', 'iwslt14_tok']:
        tokenizer = None
        if config.use_bpe:
            tokenizer = create_tokenizer(path=f'./data/{config.data.name}/')

        if config.use_mbert:
            tokenizer = AutoTokenizer.from_pretrained('/pretrained_models/bert-base-multilingual-cased')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer, clean_up_tokenization_spaces=True)
    
    if tokenizer == None:
        vocab_size = config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
        if config.data.name in ['iwslt14', 'iwslt14_tok']:
            if config.use_bpe:
                config.pad_value = tokenizer.get_vocab()['<pad>']
            # else use by fairseq
        else:
            config.pad_value = tokenizer.pad_token_id

    #region 

    # define model and diffusion
    model, diffusion = create_model(config, vocab_size), create_gaussian_diffusion(config)
    model.eval()

    # load trained model
    if config.load_from_ema:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'ema_{config.ema_rate}_checkpoint-{config.load_step}')
    else:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'model_checkpoint-{config.load_step}')
    
    # fine-tune

    model_saved_state = load_states_from_checkpoint(eval_model_path, 0)
    model.load_state_dict(model_saved_state.model_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'the parameter count is {pytorch_total_params}')

    # （）
    # if dist.get_world_size() > 1:
    #     model = DDP(
    #         model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=False,
    #     )
    # dist.destroy_process_group()

    model = model.cuda()

    print("sampling text from random noise...")
    logger.info(f"sample num is : {config.num_samples}")

    if config.ddim_sample:
        sample_fn = (diffusion.ddim_sample_loop)
    else:
        sample_fn = (diffusion.p_sample_loop)
    # TODO: add control sampling

    # （）
    # if dist.get_world_size() > 1:
    #     emb_model = model.module.word_embedding
    # else:
    emb_model = model.word_embedding

    #endregion

    # 

    if config.model.mode == 's2s':
        print(f"start generate query from dev dataset, for every passage,\
            we generate {config.num_samples} querys...")
        logger.info("***** load " + config.data.name + " dev dataset*****")
            
        if config.data.name in ['commongen']:
            dev_data = load_jsonl_data(config, 'dev')
        else:
            dev_data = load_jsonl_data(config, 'test')
        data_piece = split_data(dev_data, log=True)
        dev_dataset = S2S_dataset(data_piece, tokenizer, config, attri='test')
        
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=config.batch_size, 
            drop_last=False, pin_memory=True, num_workers=config.num_workers, 
            collate_fn=S2S_dataset.get_collate_fn(config)
        )

        logger.info(f"total query DEV dataset len : {len(dev_dataset)}")

        # 
        # random_list = random.sample(range(len(dev_dataset)), 10)
        # for idx in random_list:
        #     logger.info(f"example of {idx} is : {dev_dataset[idx]}")

        # 
        # config.num_samples
        for i in range(config.num_samples):
            # cudalist
            torch.cuda.empty_cache()
            each_sample_list = []
            json_results = []
            print(f"start sample {i+1} epoch...")
            # dev_dataloader
            for _, batch in enumerate(tqdm(dev_dataloader)):
                # ，。
                with torch.no_grad():

                    # region
                    dp_attn_masks = None
                    if config.use_src_dp:
                        # ：dp_attn_mask
                        # batchdp_mask，dp_mask[B, S, S]（s2s_dataset）
                        # [num_hidden_layer, B, num_heads, S, S]
                        dp_attn_mask = batch['src_dp_mask'].unsqueeze(1).cuda()  # boardcast for num_head dim
                        batch_size, num_head, seq_len, seq_len = dp_attn_mask.shape
                        # dp_attn_mask6
                        dp_attn_masks = torch.ones(6, batch_size, num_head, seq_len, seq_len, device=dp_attn_mask.device, dtype=dp_attn_mask.dtype)
                        # dp_attn_masks = dp_attn_mask.expand(6, batch_size, num_head, seq_len, seq_len)
                        dp_attn_masks[config.dp_head_layer] = dp_attn_mask
                        # endregion

                    encoder_hidden_states = model.encoder(
                        input_ids=batch['src_input_ids'].cuda(), 
                        attention_mask=batch['src_attention_mask'].cuda(),
                        head_mask=dp_attn_masks # ：
                    ).last_hidden_state  # [bs, seq_len, hz]

                # false
                if config.pred_len:
                    with torch.no_grad():
                        length_out = model.get_pred_len(
                            encoder_hidden_states=encoder_hidden_states,
                            src_masks=batch['src_attention_mask'].cuda(),
                            normalize=True,
                        )  # [bs, max_pos_len]
                        pred_lengs = length_out.max(-1)[1]  # [bs,], max return tuple(value, indices)

                    tgt_attention_mask = []
                    for len_item in pred_lengs:
                        tgt_attention_mask.append([1] * len_item + [0] * (max(pred_lengs) - len_item))
                    tgt_attention_mask = torch.tensor(tgt_attention_mask).long()
                    
                    input_shape = (
                        tgt_attention_mask.shape[0], tgt_attention_mask.shape[1], config.in_channels,
                    )
                else:
                    pred_lengs, tgt_attention_mask = None, None
                    input_shape = (
                        batch['src_input_ids'].shape[0], config.tgt_len, config.in_channels,
                    )

                model_kwargs = {'src_attention_mask': batch['src_attention_mask'] , # + batch['src_syntax_mask']
                                # masksrc_attention_mask；
                                # 
                                'tgt_attention_mask': tgt_attention_mask,
                                'encoder_hidden_states': encoder_hidden_states,}

                # 
                # TODO : ，、t
                sample = sample_fn(
                    model,
                    input_shape,
                    clip_denoised=config.clip_denoised,
                    # "Freeze" some parameters for easy recall.
                    denoised_fn=partial(denoised_fn_round,
                                        config, emb_model.cuda()),
                    progress=False,
                    model_kwargs=model_kwargs,
                    pred_lengs=pred_lengs,
                    top_p=-1.0,
                    # langevin_fn
                )
                # sample/hat{x_0}

                logger.debug(f"sample result shape: {sample.shape}")  # (bs, seq_len, emb_dim)
                logger.debug('decoding for e2e... ')

                logits = model.get_logits(sample)  # (bs, seq_len, vocab_size)
                sample_id_tensor = torch.argmax(logits, dim=-1)

                if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok'] and (not config.use_mbert):
                    if config.use_bpe:
                        batch_text = bpe_batch_decode(sample_id_tensor, tokenizer)
                        each_sample_list.extend(batch_text)
                    else:
                        each_sample_list.extend(sample_id_tensor.tolist())
                else:
                    each_sample_list.extend(tokenizer.batch_decode(sample_id_tensor, skip_special_tokens=True))

                if config.use_bpe:
                    src_text_list = bpe_batch_decode(batch['src_input_ids'], tokenizer)
                    tgt_text_list = bpe_batch_decode(batch['tgt_input_ids'], tokenizer)
                    pred_text_list = bpe_batch_decode(sample_id_tensor, tokenizer)
                else:
                    src_text_list = tokenizer.batch_decode(batch['src_input_ids'], skip_special_tokens=True)
                    tgt_text_list = tokenizer.batch_decode(batch['tgt_input_ids'], skip_special_tokens=True)
                    pred_text_list = tokenizer.batch_decode(sample_id_tensor, skip_special_tokens=True)
                
                for src_text, tgt_text, pred_text in zip(src_text_list, tgt_text_list, pred_text_list):
                    json_results.append({
                        'recover': pred_text,
                        'reference': tgt_text,
                        'source': src_text
                    })

            output_path = os.path.join(generate_path, 'num' + str(i))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # dist.barrier()

            out_path = os.path.join(output_path, "rank" + str(0)+"_seed_" + str(config.exp.seed) + ".txt")
            with open(out_path, 'w') as f:
                for sentence in tqdm(each_sample_list):
                    f.write(str(sentence) + '\n')
            # json output
            json_out_path = os.path.join(output_path, "rank" + str(0)+"_seed_" + str(config.exp.seed) + ".jsonl")
            logger.info(f"writing jsonl results to {json_out_path}")
            with jsonlines.open(json_out_path, 'w') as f:
                f.write_all(json_results)


    else:
        return NotImplementedError


if __name__ == "__main__":
    main()

    # # for step by step debugging
    # with initialize(version_base=None, config_path="../"):
    #     config = compose(config_name="config")
    #     config.model.name="./models/bert-base-uncased"
    #     config.exp.name="roc"
    #     config.exp.root="outputs_finetune"
    #     config.data.name="roc"
    #     config.load_step=160000
    #     config.load_from_ema=True
    #     config.max_pos_len=128
    #     config.num_samples=1
    #     config.batch_size=7
    #     config.intermediate_size=512 
    #     config.num_attention_heads=8 
    #     config.in_channels=64 
    #     config.out_channels=64 
    #     config.time_channels=64 
    #     config.skip_sample=True 
    #     config.gen_timesteps=20 
    #     config.schedule_sampler='xy_uniform' 
    #     config.time_att=True 
    #     config.att_strategy='txl' 
    #     config.tgt_len=128
    #     config.prediction=True
    #     main(config)
