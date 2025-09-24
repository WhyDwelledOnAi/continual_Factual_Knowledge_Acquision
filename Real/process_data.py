from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm, trange
import json, pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='wiki_recent')
    parser.add_argument('--model_path', type=str, default='../llama3.2-3b')
    args = parser.parse_args()
    args.model = 'llama'
    if 'qwen' in args.model_path:
        args.model = 'qwen'
    elif 'pythia' in args.model_path:
        args.model = 'pythia'
    return args

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
    return original_corpus, continual_corpus

def tokenize_cpt_test(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    original_corpus, continual_corpus = load_data(args.dataset_name)
    # continual_corpus is used to calculate fluency
    testing_tokens = []
    for example in tqdm(continual_corpus, desc='Tokenization Continual Corpus...'):
        prompt = example['prompt']
        input_ids_prompt = tokenizer(text=prompt, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
        
        for truth in example['ground_truth']:
            input_ids_answer = tokenizer(text=' '+truth, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
            
            input_ids = input_ids_prompt + input_ids_answer + [tokenizer.eos_token_id]
            info = {'input_ids': input_ids, 'idx':len(input_ids_prompt)}
            testing_tokens.append(info)
    with open(f'{args.dataset_name}/cpt_test-{args.model}.pkl', 'wb') as f:
        pickle.dump(testing_tokens, f)
        
    testing_tokens = []
    for example in tqdm(original_corpus, desc='Tokenization Original Corpus...'):
        prompt = example['prompt']
        input_ids_prompt = tokenizer(text=prompt, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
        
        for truth in example['ground_truth']:
            input_ids_answer = tokenizer(text=' '+truth, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
            
            input_ids = input_ids_prompt + input_ids_answer + [tokenizer.eos_token_id]
            info = {'input_ids': input_ids, 'idx':len(input_ids_prompt)}
            testing_tokens.append(info)
    with open(f'{args.dataset_name}/pt_test-{args.model}.pkl', 'wb') as f:
        pickle.dump(testing_tokens, f)
            
def tokenize_cpt_train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    with open(f'{args.dataset_name}/augmented_continual.json', 'r') as f:
        training_corpus = json.load(f)
    
    training_tokens = []
    for example in tqdm(training_corpus, desc='Tokenization Training Corpus...'):
        prompt = example['prompt']
        input_ids_prompt = tokenizer(text=prompt, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
        
        for truth in example['ground_truth']:
            input_ids_answer = tokenizer(text=' '+truth, 
                padding=False, truncation=False, 
                add_special_tokens=True, return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
            input_ids = input_ids_prompt + input_ids_answer + [tokenizer.eos_token_id]
            info = {'input_ids': input_ids, 'idx':len(input_ids_prompt)}
            training_tokens.append(info)
    with open(f'{args.dataset_name}/cpt_train-{args.model}.pkl', 'wb') as f:
        pickle.dump(training_tokens, f)
        

if __name__ == '__main__':
    args = parse_args()
    # tokenize_cpt_test(args)
    tokenize_cpt_train(args)

