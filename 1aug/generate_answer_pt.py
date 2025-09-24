from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm, trange
import pandas as pd
import argparse
import pickle, json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pythia-160m')
    parser.add_argument('--model_path', type=str, 
        default='outputs/pt-pythia/final_model')
    
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--gpu_percent', type=float, default=0.8)
    args = parser.parse_args()
    return args
args = parse_args()
args.output_path = f'data_{args.model}'

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

def load_model():
    model = LLM(args.model_path, 
        gpu_memory_utilization=args.gpu_percent, 
        dtype='bfloat16', 
        max_model_len=512) 
    sampling_params = SamplingParams(
        n=1,
        logprobs=1, # obtain the probabilities of the generated responses
        seed=2025,
        temperature=args.temperature, # sampling temperature, make sure not larger than 1.0
        max_tokens=32, # generating attribute needs few tokens
    )
    return model, sampling_params

def load_prompts():
    profile = pd.read_parquet(f'data_{args.model}/profile.parquet')
    keep_cols = ['Fullname', 'Birthdate', 'Birthplace', 'University', 'Major', 'Company']
    profile = profile[keep_cols]
    name2attr = profile.set_index('Fullname').T.to_dict()
    
    biographies = pd.read_parquet(f'data_{args.model}/{args.mode}_bio.parquet')

    prompts = []
    for i in trange(len(biographies)):
        row = biographies.iloc[i]
        fullname = row['Fullname']
        attr_dict = name2attr[fullname]
        attrs = list(attr_dict.values())
        find_attr_type = {v: k for k, v in attr_dict.items()}
        
        biography = row['Content']
        statements = biography.split(fullname)[1:]
        for statement in statements:
            for attr in attrs:
                attr_idx = statement.find(attr)
                if attr_idx == -1:
                    continue
                template = statement[:attr_idx]
                prompts.append({
                    "Fullname":fullname,
                    "Number": row['Number'],
                    "Template": template,
                    "Attr_type": find_attr_type[attr],
                    "Answer": attr,
                    "Prompt": ' '+fullname+template,
                    "Prediction": None
                })
    prompts = pd.DataFrame(prompts)
    prompts.to_parquet(f'{args.output_path}/{args.mode}_prompt.parquet')
    return prompts

def load_prompts2():
    if os.path.exists(f'{args.output_path}/{args.mode}_prompt.parquet'):
        print(f"Load prompt from {args.output_path}/{args.mode}_prompt.parquet")
        prompts = pd.read_parquet(f'{args.output_path}/{args.mode}_prompt.parquet')
        return prompts
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=512,
        padding_side="right", 
        use_fast=True,)
    Attributes = ['Birthdate', 'Birthplace', 'University', 'Major', 'Company']
    
    with open(f'data_{args.model}/pt_{args.mode}.pkl', 'rb') as f:
        all_tokens = pickle.load(f)
    
    
    prompts = []
    for example in tqdm(all_tokens):
        try:
            input_ids = example['input_ids']

            index1_list = [example[f'{attr} idx'] for attr in Attributes]
            index1_list.sort()
            index2attr = {example[f'{attr} idx']:attr for attr in Attributes}
            
            start = 0
            for index in index1_list:
                fullname = example['Fullname']
                number = example['Number']
                
                attr = index2attr[index]
                index2 = example[f'{attr} idx2']
                
                prompt_tokens = input_ids[start:index]
                answer_tokens = input_ids[index:index2]
                
                prompt = tokenizer.decode(prompt_tokens)
                answer = tokenizer.decode(answer_tokens)
                template = prompt.replace(fullname, "[Name]")
                if answer[0] == ' ':
                    answer = answer[1:]
                
                start = index2 + 1
                prompts.append({
                        "Fullname":fullname,
                        "Number": number,
                        "Template": template,
                        "Attr_type": attr,
                        "Answer": answer,
                        "Prompt": fullname+template,
                        "Prediction": None
                    })
        except:
            continue
    prompts = pd.DataFrame(prompts)
    prompts.to_parquet(f'{args.output_path}/{args.mode}_prompt.parquet')
    return prompts


def generate(prompts, model, sampling_params):
    prompts_list = prompts['Prompt'].tolist()
    print("Start Generating...")
    responses = model.generate(prompts_list, sampling_params)
    
    texts =[response.outputs[0].text for response in responses]
    if len(texts) < len(prompts):
        texts += [None] * (len(prompts)-len(texts))
    prompts['Prediction'] = texts
    prompts.to_parquet(f'{args.model_path}/{args.mode}_response.parquet')
    
    
def calc_EM():
    answers = pd.read_parquet(f'{args.model_path}/{args.mode}_response.parquet')
    answers['Match'] = answers.apply(lambda row: row['Answer'] in row['Prediction'], axis=1)
    print(answers.head(5))
    print(answers['Match'].value_counts())
    print("Accuracy:", answers['Match'].mean())

    
if __name__ == '__main__':
    if not os.path.exists(f'{args.model_path}/{args.mode}_response.parquet'):
        prompts = load_prompts2()
        model, sampling_params = load_model()
        generate(prompts, model, sampling_params)
        print('Generation finished.')
    calc_EM()