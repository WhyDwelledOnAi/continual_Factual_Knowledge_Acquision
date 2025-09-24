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

class QADataset(Dataset):
    """
        Used for training in the SFT stage.
        CrossEntropy is calculated for answer token.
        Note that chunking is not used, padding is used instead.
    """
    def __init__(self,
                 data_path_list: list[str]=["data/finetune_train_token.pkl"],
                 data_ratio_list: list[float]=[1.0],
                 tokenizer: transformers.PreTrainedTokenizer=None,
                 ):
        super(QADataset, self).__init__()
        
        self.raw_data = []
        for data_path, data_ratio in zip(data_path_list, data_ratio_list):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            data = random.sample(data, int(len(data) * data_ratio))
            self.raw_data.extend(data)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=512,
                padding_side="right",
                use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_length = 32

        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        for info in tqdm(self.raw_data, desc='Constructing training QA Dataset'):
            question_ids = info['question_ids']
            answer_ids = info['answer_ids']
            pad_length = self.max_length - (len(question_ids) + len(answer_ids) + 1) # +1 for <eos>
            input_ids = torch.tensor(
                question_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
                dtype=torch.int64
            )
            target_ids = torch.tensor(
                [IGNORE_TOKEN_ID] * len(question_ids) + answer_ids + [tokenizer.eos_token_id] +
                [IGNORE_TOKEN_ID] * pad_length,
                dtype=torch.int64
            )
            # although the token indicating <eos> will also be masked, it is not a problem
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            self.input_ids_list.append(input_ids)
            self.label_list.append(target_ids)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }
class QAFirstTokenAccuracyDataset(Dataset):
    """
        Used for testing in the fine-tuning stage.
        test the accuracy of the attributes first token prediction.
        So no shuffle is needed.
    """
    def __init__(self,
            data_path: str = "data/finetune_train_token.pkl",
            tokenizer: transformers.PreTrainedTokenizer = None,
            ):  
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.attributes = ["Birthdate", "Birthplace", 
            "University", "Major", "Company",]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/new_disk2/haoyu_wang/LLMs/pythia-160m",
                model_max_length=512,
                padding_side="right",
                use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_length = 32
        
        self.input_ids_list, self.answer_id_list = [], []
        self.attention_mask_list, self.att_list = [], []
        for info in tqdm(self.raw_data, desc='Constructing testing QA dataset.'):
            question_ids = info['question_ids']
            answer_first_ids = info['answer_ids'][0]
            pad_length = self.max_length - len(question_ids) 
            input_ids = torch.tensor(
                [tokenizer.pad_token_id] * pad_length + question_ids,
                dtype=torch.int64
            ) # left padding for inference
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            self.input_ids_list.append(input_ids)
            self.answer_id_list.append(answer_first_ids)
            self.attention_mask_list.append(attention_mask)
            self.att_list.append(info['attribute'])

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'answer_id': self.answer_id_list[idx],
            'attention_mask': self.attention_mask_list[idx],
            'attribute': self.att_list[idx]
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "../pythia-160m",
        model_max_length=512,
        padding_side="right",
        use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
