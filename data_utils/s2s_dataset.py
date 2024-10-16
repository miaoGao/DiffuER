import os
import torch
import datasets
import random
import logging
import jsonlines
import pickle

# import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from data_utils.preprocess_story import get_examples
# from torch.utils.data.distributed import DistributedSampler

# from data_utils.fairseq_dataset import load_fairseq

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def edge_index_to_adj_mat(edge_index, size):
    square_mat = torch.zeros(size, size, dtype=torch.long)
    square_mat[edge_index[:, 0], edge_index[:, 1]] = 1
    return square_mat

def load_jsonl_data(config, attri):
    
    config.fairseq.use_fairseq = False
    data = []

    # stanza，
    if config.use_stanza == True:
        print("stanza")

        if config.data.name in ['cnn_dm', 'xsum', 'gigaword', 'squad', 'personachat', 'coqa']:
            src_path = os.path.join(config.data.path, config.data.name, attri + '.src')
            tgt_path = os.path.join(config.data.path, config.data.name, attri + '.tgt')

            src_data = open(src_path, 'r')
            tgt_data = open(tgt_path, 'r')
            for src, tgt in zip(src_data, tgt_data):
                data.append({
                    'src': src.strip('\n'),
                    'tgt': tgt.strip('\n'),
                })
            src_data.close()
            tgt_data.close()


    # stanza，    
    else:
        if config.data.name == 'commongen':
            path = os.path.join(config.data.path, config.data.name, attri + '.jsonl')

            with jsonlines.open(path, 'r') as rp:
                for line in rp:
                    for item in line['scene']:
                        data.append({
                            'src': line['concept_set'],
                            'tgt': item
                        })
            rp.close()
        
        
        elif config.data.name in ['qqp', 'quasar_t', 'text_simple']:
            if attri == 'dev':
                attri = 'valid'
            path = os.path.join(config.data.path, config.data.name, attri + '.jsonl')

            with jsonlines.open(path, 'r') as rp:
                for item in rp:
                    data.append({
                        'src': item['src'],
                        'tgt': item['trg'],
                    })
            rp.close()

        elif config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
            if attri == 'dev':
                if config.data.name == 'wmt14_hug':
                    attri = 'validation'
                else:
                    attri = 'valid'
                
            if config.use_bpe:
                if config.data.name == 'wmt14_hug':
                    path = os.path.join(config.data.path, config.data.name)
                    raw_data = datasets.load_from_disk(path)[attri]
                    for item in tqdm(raw_data['translation'], desc=attri):
                        data.append({
                            'src': item[config.src_lang].strip('\n'),
                            'tgt': item[config.tgt_lang].strip('\n')
                        })
                else:
                    path = os.path.join(config.data.path, config.data.name, attri)
                    raw_data = [open(path+'.'+config.src_lang, 'r').readlines(),
                                open(path+'.'+config.tgt_lang, 'r').readlines()]
                    
                    for src, tgt in zip(raw_data[0], raw_data[1]):
                        data.append({
                            'src': src.strip('\n'),
                            'tgt': tgt.strip('\n')
                        })
            else:
                path = os.path.join(config.data.path, config.data.name, attri)
                raw_data = [open(path+'.'+config.src_lang, 'r').readlines(),
                            open(path+'.'+config.tgt_lang, 'r').readlines()]
                
                for src, tgt in zip(raw_data[0], raw_data[1]):
                    data.append({
                        'src': src.strip('\n'),
                        'tgt': tgt.strip('\n')
                    })

            
        elif config.data.name in ['cnn_dm', 'xsum', 'gigaword', 'squad', 'personachat', 'coqa']:
            src_path = os.path.join(config.data.path, config.data.name, attri + '.src')
            tgt_path = os.path.join(config.data.path, config.data.name, attri + '.tgt')

            src_data = open(src_path, 'r')
            tgt_data = open(tgt_path, 'r')
            for src, tgt in zip(src_data, tgt_data):
                data.append({
                    'src': src.strip('\n'),
                    'tgt': tgt.strip('\n'),
                })
            src_data.close()
            tgt_data.close()


    if config.data.name == 'writing_prompt':
        examples = get_examples(config.data.name, attri)
        for e in examples:
            data.append({
                'src': e.src_sent,
                'tgt': e.tgt_sent
            })

    if config.data.name == "roc":
        
        path = os.path.join(config.data.path, config.data.name, attri + '.jsonl')

        with jsonlines.open(path, 'r') as f:
            for line in f:
                data.append({
                'src': line['src'],
                'tgt': line['trg']
                })

    random_list = random.sample(range(len(data)), 5)
    for idx in random_list:
        logger.info(f"example of {idx} is : {data[idx]}")

    return data   # TODO: 


def load_s2s_data(config, tokenizer):
    logger.info("***** load " + config.data.name + " train dataset *****")
        
    # if 'C4' in config.data.name:
    #     data = datasets.load_from_disk(
    #         os.path.join(config.data.path, config.data.name, 'train')
    #     )
    # else:
    train_data = load_jsonl_data(config, attri='train')
    train_dataset = S2S_dataset(train_data, tokenizer, config, attri='train')

    logger.info("***** load " + config.data.name + " dev dataset *****")
    # if 'C4' in config.data.name:
    #     data = datasets.load_from_disk(
    #         os.path.join(config.data.path, config.data.name, 'dev')
    #     )
    # else:
    if config.data.name == "commongen":
        dev_data = load_jsonl_data(config, attri='dev')
    else:
        dev_data = load_jsonl_data(config, attri='test')
    dev_dataset = S2S_dataset(dev_data, tokenizer, config, attri='test')
    
    # logger.info(f"example of TRAIN id lists: {train_dataset[50]}")
    logger.info(f"total query TRAIN dataset len : {len(train_dataset)}")
    # logger.info(f"example of DEV id lists: {dev_dataset[50]}")
    logger.info(f"total query DEV dataset len : {len(dev_dataset)}")
    
    # train_sample = DistributedSampler(train_dataset)
    # sampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size, drop_last=False, num_workers=config.num_workers, 
        collate_fn=S2S_dataset.get_collate_fn(config)
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        drop_last=False, pin_memory=True, num_workers=config.num_workers, 
        collate_fn=S2S_dataset.get_collate_fn(config)
    )

    return train_dataloader, dev_dataloader


class S2S_dataset(Dataset):
    def __init__(self, data, tokenizer, config, attri='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.dp_data = None

        if self.config.use_src_dp:
            if attri == 'train':
                logger.info("loadding xsum train src dp data")
                self.dp_data = pickle.load(open('data/xsum/train_dp.bin', 'rb'))
            if attri == 'test':
                logger.info("loadding xsum test src dp data")
                self.dp_data = pickle.load(open('data/xsum/test_dp.bin', 'rb'))

    def __getitem__(self, index):
        example = self.data[index]
        
        if self.config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok'] and (not self.config.use_mbert):
            if self.config.use_bpe:
                src_input_ids = torch.LongTensor(self.tokenizer.encode(example['src']).ids)
                tgt_input_ids = torch.LongTensor(self.tokenizer.encode(example['tgt']).ids)
            else:
                src_input_ids = example['src']
                tgt_input_ids = example['tgt']

        else:
            src_input_ids = self.tokenizer.encode(example['src'],
                                                    padding='longest',
                                                    truncation=True,
                                                    max_length=self.config.max_pos_len,
                                                    return_tensors='pt')
            tgt_input_ids = self.tokenizer.encode(example['tgt'],
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=self.config.tgt_len,
                                                    return_tensors='pt')

        if self.config.use_src_dp:
            if 0 <= index <= len(self.dp_data):
                curr_dp_data = self.dp_data[index]
                edge_index, size = curr_dp_data['edge_index'], curr_dp_data['size']
                attn_mask = edge_index_to_adj_mat(edge_index, size)
                seq_len = attn_mask.shape[0] + 2
                # padding
                padded_attn_mask = torch.ones(seq_len, seq_len, dtype=torch.long)
                padded_attn_mask[1:-1, 1:-1] = attn_mask

            return {'src': src_input_ids, 'tgt': tgt_input_ids, 'dp_attn_mask': padded_attn_mask}
        else:
            return {'src': src_input_ids, 'tgt': tgt_input_ids}
    
    # region
    
    def decode_token_list(self, token_list):
        decoded_text = ""
        for token in token_list:
            if token.startswith("##"):
                decoded_text += token[2:]
            else:
                decoded_text += " " + token
        return decoded_text.strip()
    
    # endregion

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_collate_fn(cls, config):
        def fn(batch):
            src, tgt, length, dp_masks = [], [], [], []
            dp_mask_batch = None
            
            for item in batch:
                src.append(item['src'].squeeze())
                tgt.append(item['tgt'].squeeze())
                length.append(min(len(item['tgt'].squeeze()), config.tgt_len))
                
                if config.use_src_dp:
                    dp_masks.append(item['dp_attn_mask'])

            if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok', 'roc']:
                # and (not config.use_mbert):
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                src_tensor = src_tensor[:, :config.max_pos_len]
                tgt_tensor = pad_sequence(tgt, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = tgt_tensor[:, :config.tgt_len]
                if not config.pred_len and config.tgt_len > tgt_tensor.size(1):
                    # padding to max target length
                    tgt_tensor = torch.cat((tgt_tensor, torch.tensor(config.pad_value).repeat(
                        tgt_tensor.size(0), (config.tgt_len - tgt_tensor.size(1)))), dim=-1) 
            else:
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = torch.stack(tgt)
                
                if config.use_src_dp:
                    batch_size = src_tensor.shape[0]
                    padded_seq_len = src_tensor.shape[-1]
                    dp_mask_batch = torch.zeros(batch_size, padded_seq_len, padded_seq_len, dtype=torch.long)
                    for i, dp_mask in enumerate(dp_masks):
                        seq_len = dp_mask.shape[0]
                        dp_mask_batch[i, :seq_len, :seq_len] = dp_mask

            if config.pred_len:
                length_tensor = torch.tensor(length).long()
                tgt_attention_mask = (tgt_tensor != config.pad_value).long()
            else:
                length_tensor = torch.tensor(config.tgt_len).repeat(tgt_tensor.size(0)).long()
                tgt_attention_mask = torch.ones_like(tgt_tensor).long()

            if config.prediction:
                return {"src_input_ids": src_tensor.long(), 
                        "src_attention_mask": (src_tensor != config.pad_value).long(),
                        "tgt_input_ids": tgt_tensor,
                        "src_dp_mask": dp_mask_batch
                        }
            else:
                return {"src_input_ids": src_tensor.long(), 
                        "src_attention_mask": (src_tensor != config.pad_value).long(),
                        "src_dp_mask": dp_mask_batch,
                        "tgt_input_ids": tgt_tensor, 
                        "tgt_attention_mask": tgt_attention_mask,
                        "length": length_tensor,}
           
        return fn
    
    # ，generate.py
    def get_collate_fn_origin(cls, config):
        def fn(batch):
            src, tgt, length = [], [], []
            for item in batch:
                src.append(item['src'].squeeze())
                tgt.append(item['tgt'].squeeze())
                length.append(min(len(item['tgt'].squeeze()), config.tgt_len))

            if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
                # and (not config.use_mbert):
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                src_tensor = src_tensor[:, :config.max_pos_len]
                tgt_tensor = pad_sequence(tgt, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = tgt_tensor[:, :config.tgt_len]
                if not config.pred_len and config.tgt_len > tgt_tensor.size(1):
                    # padding to max target length
                    tgt_tensor = torch.cat((tgt_tensor, torch.tensor(config.pad_value).repeat(
                        tgt_tensor.size(0), (config.tgt_len - tgt_tensor.size(1)))), dim=-1)
                
            else:
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = torch.stack(tgt)

            if config.pred_len:
                length_tensor = torch.tensor(length).long()
                tgt_attention_mask = (tgt_tensor != config.pad_value).long()
            else:
                length_tensor = torch.tensor(config.tgt_len).repeat(tgt_tensor.size(0)).long()
                tgt_attention_mask = torch.ones_like(tgt_tensor).long()

            if config.prediction:
                return {"src_input_ids": src_tensor.long(), 
                        "src_attention_mask": (src_tensor != config.pad_value).long(),}
            else:
                return {"src_input_ids": src_tensor.long(), 
                        "src_attention_mask": (src_tensor != config.pad_value).long(),
                        "tgt_input_ids": tgt_tensor, 
                        "tgt_attention_mask": tgt_attention_mask,
                        "length": length_tensor,}
        return fn