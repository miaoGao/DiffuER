import hydra
import torch
import logging

from transformers import set_seed, AutoTokenizer

# from train_utils.pretrain import PretrainLoop
from train_utils.trainer import TrainLoop
from utils import load_states_from_checkpoint
from model_utils.create_model import create_model, create_gaussian_diffusion
from train_utils.resample import create_named_schedule_sampler
from data_utils.s2s_dataset import load_s2s_data
from data_utils.tokenizer_utils import create_tokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(config):
    local_rank = 0

    config.exp.dir = os.path.join(config.exp.root, config.data.name, config.exp.name)
    if (local_rank == 0) and (not os.path.exists(config.exp.dir)):
        os.makedirs(config.exp.dir)

    # torch.cuda.set_device("cuda:0")  # ddp setting
    # dist.init_process_group(backend="nccl")  # ddp setting
    
    set_seed(config.exp.seed)  # seed setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.visible_device)
    
    # load tokenizer
    if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
        tokenizer = None
        if config.use_bpe:
            tokenizer = create_tokenizer(path=f'./data/{config.data.name}/')
        elif config.use_mbert:
            tokenizer = AutoTokenizer.from_pretrained('/pretrained_models/bert-base-multilingual-cased')
    else:
        if config.use_sentence_piece:
            tokenizer = AutoTokenizer.from_pretrained('t5-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
            
    if tokenizer == None:
        vocab_size = config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
        if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
            if config.use_bpe:
                config.pad_value = tokenizer.get_vocab()['<pad>']
            # else use by fairseq
        else:
            config.pad_value = tokenizer.pad_token_id

    print(config)

    # load model (predict Guassian noise) and diffusion basic class
    model, diffusion = create_model(config, vocab_size), create_gaussian_diffusion(config)
    if config.model.pretrain is not None:
        # if dist.get_rank() == 0:
        logger.info(f"load model ckpt at : {config.model.pretrain}")
        saved_state = load_states_from_checkpoint(config.model.pretrain, 0)
        model.load_state_dict(saved_state.model_dict, strict=False)
    model.to(config.device)

    # calculate the total parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'the parameter count is {pytorch_total_params}')

    # uniform / loss-second-moment
    schedule_sampler = create_named_schedule_sampler(config, diffusion)

    # load data
    print("creating data loader...")
    
    # 
    train_dataloader, dev_dataloader = load_s2s_data(config, tokenizer=tokenizer)

    # for batch in train_dataloader:
    #     print(tokenizer.batch_decode(batch['src_input_ids'], skip_special_tokens=False))
    #     print(tokenizer.batch_decode(batch['tgt_input_ids'], skip_special_tokens=False))
    #     break

    # training section
    print("training Diffusion LM model...")
        
    # if config.data.name == 'pretrain':
    #     PretrainLoop(
    #         config=config,
    #         tokenizer=tokenizer,
    #         model=model,
    #         diffusion=diffusion,
    #         data=train_dataloader,
    #         dev_data=dev_dataloader,
    #         schedule_sampler=schedule_sampler,
    #     ).run_loop()
    # else:
    TrainLoop(
        config=config,
        model=model,
        diffusion=diffusion,
        data=train_dataloader,
        dev_data=dev_dataloader,
        schedule_sampler=schedule_sampler,
    ).run_loop()


if __name__ == "__main__":
    main()
