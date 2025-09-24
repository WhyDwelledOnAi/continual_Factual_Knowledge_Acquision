import random
import json, pickle
from tqdm import trange, tqdm
from functools import reduce

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def load_data(dataset_name):
    assert dataset_name in ['zsre', 'wiki_bio', 'wiki_recent'],\
        f"Dataset {dataset_name} is not supported."
    
    with open(f'{dataset_name}/test.json') as f:
        data = json.load(f)
    
    original_corpus = []
    continual_corpus = []
    for example in tqdm(data, desc=f'Load {dataset_name} data...'):
        continual_samples = []
        if dataset_name == 'zsre':
            continual_samples.append(
                {'prompt': example['prompt'],
                'ground_truth': example['ground_truth'],
                }
            )
            if 'rephrase_prompt' in example.keys():
                continual_samples.append(
                {'prompt': example['rephrase_prompt'],
                'ground_truth': example['ground_truth'],
                }
            )
        elif dataset_name == 'wiki_bio':
            continual_samples.append(
                {'prompt': example['text'],
                'ground_truth': example['labels'],
                }
            )
        elif dataset_name == 'wiki_recent':
            continual_samples.append(
                {'prompt': example['prompt'],
                'ground_truth': example['target_new'],
                }
            )
            if 'rephrase' in example.keys():
                continual_samples.append(
                {'prompt': example['rephrase'],
                'ground_truth': example['target_new'],
                }
            )
        continual_corpus.extend(continual_samples)
        
        original_samples = []
        if dataset_name == 'zsre':
            original_samples.extend(example['locality']['Relation_Specificity'])
        elif dataset_name == 'wiki_bio':
            original_samples.extend(example['locality']['Relation_Specificity'])
        elif dataset_name == 'wiki_recent':
            for item in example['locality'].values():
                for original_sample in item:
                    original_samples.append(
                        {'prompt': original_sample['prompt'],
                        'ground_truth': original_sample['ground_truth'][0]
                        })
        original_corpus.extend(original_samples)
    print(len(original_corpus), len(continual_corpus))
    return original_corpus, continual_corpus


class PretrainDataset(Dataset):
    def __init__(self,
            data_path_list:list[str] = ["data/pt_train.pkl",],
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):
        super(PretrainDataset, self).__init__()
        
        self.raw_data = [] # list[ list[int] ]
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                new_data = pickle.load(f)
            input_ids_list = [item['input_ids'] for item in new_data]
            self.raw_data.extend(input_ids_list)

        # load tokenizer for using tokenizer.eos_token later
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=512,
                padding_side="right",
                use_fast=True)
        self.tokenizer = tokenizer
        self.attention_mask = torch.tensor([True] * tokenizer.model_max_length, dtype=torch.bool)
        
        self.input_ids_list, self.label_list = [], []
        # 在每个epoch开始时重新打乱数据并chunking
        self.construct_dataset()

    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list = [], []
        random.shuffle(self.raw_data)
        flattened_tokens = []
        for item in self.raw_data:
            flattened_tokens.extend(item)
        for i in trange(0, len(flattened_tokens), 
                        self.tokenizer.model_max_length, 
                        desc='Training biography data chunking'):
            ids_chunk = flattened_tokens[i:i + self.tokenizer.model_max_length]
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else: # padding for the last chunk
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.eos_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            # the first token of each sentence is always masked
            target_mask = torch.cat((torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.eos_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)
            
            self.input_ids_list.append(input_ids)
            self.label_list.append(target)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask,
        }

class MaskedBiographyDataset(Dataset):
    """
        Used for training in the fine-tune stage.
        CrossEntropy is calculated ONLY for attribute token.
        Note that chunking is used.
    """
    def __init__(self,
            data_path_list:list = [
                "data/pretrain_train_token.pkl", 
            ],
            data_ratio_list:list = [1.0 ,],
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):
        self.raw_data = []
        for data_path, data_ratio in zip(data_path_list, data_ratio_list):
            with open(data_path, 'rb') as f:
                new_data = pickle.load(f)
            new_data = random.sample(new_data, int(len(new_data) * data_ratio))
            self.raw_data.extend(new_data)
        self.attributes = ["Birthdate", "Birthplace", 
            "University", "Major", "Company",]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=512,
                padding_side="right",
                use_fast=True)
        self.tokenizer = tokenizer

        self.attention_mask = torch.tensor([True] * tokenizer.model_max_length, dtype=torch.bool)
        self.input_ids_list, self.label_list = [], []
        self.construct_dataset()
    
    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list = [], []
        random.shuffle(self.raw_data)
        # flattening for chunking
        flattened_input_ids_list, flattened_label_list = [], []
        for info in self.raw_data:
            input_ids = info['input_ids']
            label = [IGNORE_TOKEN_ID] * len(input_ids)
            for attribute in self.attributes:
                first_token_index = info[attribute+' idx']
                last_token_index = info[attribute+' idx2']
                label[first_token_index:last_token_index] = input_ids[first_token_index:last_token_index]
            flattened_input_ids_list.extend(input_ids)
            flattened_label_list.extend(label)
        # chunking
        for i in trange(0, len(flattened_input_ids_list), 
                tokenizer.model_max_length, 
                desc='Testing biography data chunking'):
            ids_chunk = flattened_input_ids_list[i:i + tokenizer.model_max_length]
            label_chunk = flattened_label_list[i:i + tokenizer.model_max_length]
            if len(ids_chunk) == tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                label = torch.tensor(label_chunk, dtype=torch.int64)
            else:
                input_ids = torch.tensor(
                    ids_chunk + [tokenizer.eos_token_id] * (tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                label = torch.tensor(
                    label_chunk + [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            label[0] = IGNORE_TOKEN_ID
            self.input_ids_list.append(input_ids)
            self.label_list.append(label)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask,
        }

class FirstTokenAccuracyDataset(Dataset):
    """
        Used for testing in the pre-training stage.
        test the accuracy of the attributes first token prediction.
        So no shuffle is needed.
    """
    def __init__(self,
            data_path_list: list[str] = ["data/pt_test.pkl", ],
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):  
        super(FirstTokenAccuracyDataset, self).__init__()
        self.raw_data = [] # list[ dict]
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                new_data = pickle.load(f)
            self.raw_data.extend(new_data)
        
        self.attributes = ["Birthdate", "Birthplace", "University", "Major", "Company",]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=512,
                padding_side="right",
                use_fast=True)
        self.tokenizer = tokenizer
        self.attention_mask = torch.tensor([True] * tokenizer.model_max_length, dtype=torch.bool)
        
        self.input_ids_list = []
        self.token_position_list = []

        flattened_input_ids_list = []
        flattened_token_position_list = []
        for info in self.raw_data:
            input_ids = info['input_ids']
            first_token_position = [None] * len(input_ids)
            for attribute in self.attributes:
                first_token_index = info[attribute+' idx']
                first_token_position[first_token_index] = (attribute, input_ids[first_token_index])
            flattened_input_ids_list.extend(input_ids)
            flattened_token_position_list.extend(first_token_position)
        # chunking
        for i in trange(0, len(flattened_input_ids_list), 
                tokenizer.model_max_length, 
                desc='Testing biography data chunking'):
            ids_chunk = flattened_input_ids_list[i:i + tokenizer.model_max_length]
            pst_chunk = flattened_token_position_list[i:i + tokenizer.model_max_length]
            if len(ids_chunk) == tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
            else:
                input_ids = torch.tensor(
                    ids_chunk + [tokenizer.eos_token_id] * (tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            token_position = {}
            for attribute in self.attributes:
                token_position[attribute] = {
                    'index': torch.tensor([False] * len(input_ids), dtype=torch.bool),
                    'first_token_list': []
                }
            for info_id in range(len(pst_chunk)):
                info = pst_chunk[info_id]
                if info is None:
                    continue
                attribute, first_token = info
                assert input_ids[info_id] == first_token
                token_position[attribute]['index'][info_id] = True
                token_position[attribute]['first_token_list'].append(first_token)
            self.input_ids_list.append(input_ids)
            self.token_position_list.append(token_position)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'token_position': self.token_position_list[idx],
            'attention_mask': self.attention_mask,
        }

class CPTDataset(Dataset):
    def __init__(self,
            data_path_list:list[str] = ['cpt_train-pythia.pkl'],
            data_ratio_list:list[float] = [1.0],
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):
        super(CPTDataset, self).__init__()
        
        self.target_sample_num = 0
        self.raw_data = [] 
        for i, (data_path, data_ratio) in enumerate(zip(data_path_list, data_ratio_list)): 
            with open(data_path, 'rb') as f: 
                new_data = pickle.load(f) 
            sample_num = len(new_data)
            if i == 0:
                self.target_sample_num = sample_num / data_ratio
            target_sample_num = int(self.target_sample_num * data_ratio)
            new_data_resampled = random.choices(new_data, k=target_sample_num)
            input_ids_list = [item['input_ids'] for item in new_data_resampled] 
            self.raw_data.extend(input_ids_list)
            

        # load tokenizer for using tokenizer.eos_token later
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=256,
                padding_side="right",
                use_fast=True)
        self.tokenizer = tokenizer
        self.attention_mask = torch.tensor([True] * tokenizer.model_max_length, dtype=torch.bool)
        
        self.input_ids_list, self.label_list = [], []
        # 在每个epoch开始时重新打乱数据并chunking
        self.construct_dataset()
        
    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list = [], []
        random.shuffle(self.raw_data)
        flattened_tokens = []
        for item in self.raw_data:
            flattened_tokens.extend(item)
        for i in trange(0, len(flattened_tokens), 
                        self.tokenizer.model_max_length, 
                        desc='Training biography data chunking'):
            ids_chunk = flattened_tokens[i:i + self.tokenizer.model_max_length]
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else: # padding for the last chunk
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.eos_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            # the first token of each sentence is always masked
            target_mask = torch.cat((torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.eos_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)
            
            self.input_ids_list.append(input_ids)
            self.label_list.append(target)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask,
        }


class FluencyDataset(Dataset):
    def __init__(self,
            data_path_list: list[str] = ["data/pt_test.pkl"],
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):  
        super(FluencyDataset, self).__init__()
        self.raw_data = [] # list[ dict]
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                new_data = pickle.load(f)
            self.raw_data.extend(new_data)
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=256,
                padding_side="right",
                use_fast=True)
        self.tokenizer = tokenizer
        self.attention_mask = torch.tensor([True] * tokenizer.model_max_length, dtype=torch.bool)
        
        self.input_ids_list = []
        self.label_ids_list = []

        flattened_input_ids_list = []
        flattened_label_ids_list = []
        for info in self.raw_data:
            input_ids = info['input_ids']
            first_token_index = info['idx']
            
            label_ids = [IGNORE_TOKEN_ID] * (first_token_index) + input_ids[info['idx']:]
            flattened_input_ids_list.extend(input_ids)
            flattened_label_ids_list.extend(label_ids)
        
        # chunking
        for i in range(0, len(flattened_input_ids_list), 
                tokenizer.model_max_length):
        # for i in trange(0, len(flattened_input_ids_list), 
        #         tokenizer.model_max_length, 
        #         desc='Testing biography data chunking'):
            input_ids_chunk = flattened_input_ids_list[i:i + tokenizer.model_max_length]
            label_ids_chunk = flattened_label_ids_list[i:i + tokenizer.model_max_length]
            if len(input_ids_chunk) == tokenizer.model_max_length:
                input_ids = torch.tensor(input_ids_chunk, dtype=torch.int64)
                label_ids = torch.tensor(label_ids_chunk, dtype=torch.int64)
            else:
                input_ids = torch.tensor(
                    input_ids_chunk + [tokenizer.eos_token_id] * (tokenizer.model_max_length - len(input_ids_chunk)),
                    dtype=torch.int64)
                label_ids = torch.tensor(
                    label_ids_chunk + [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(label_ids_chunk)),
                    dtype=torch.int64) 
            
            
            self.input_ids_list.append(input_ids)
            self.label_ids_list.append(label_ids)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'label_ids': self.label_ids_list[idx],
            'attention_mask': self.attention_mask,
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "../pythia-160m",
        model_max_length=512,
        padding_side="right",
        use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # cpt_dataset = CPTDataset(
    #     data_path_list=['wiki_bio/cpt_rehersal_stoc-pythia.pkl', 'wiki_bio/cpt_train-pythia.pkl'],
    #     data_ratio_list=[0.2, 0.8],
    #     tokenizer=tokenizer)
    cpt_test_dataset = FluencyDataset(
        data_path_list=['wiki_bio/cpt_test-pythia.pkl'],
        tokenizer=tokenizer
    )
    for i in trange(0, len(cpt_test_dataset), 
                48, 
                desc='Calculating first token accuracy'):
            batch = cpt_test_dataset[i: i + 48]
            
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to('cuda')
            labels_ids = torch.cat([ids.unsqueeze(0) for ids in batch['label_ids']], dim=0)
            labels_ids = labels_ids.to('cuda')
            shift_labels = labels_ids[:, 1:]  # [bs, seq_len-1]
            print(shift_labels.shape)
            
            mask = labels_ids != IGNORE_TOKEN_ID
            
    