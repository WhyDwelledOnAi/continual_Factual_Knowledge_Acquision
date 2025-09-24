import torch
import os, json, pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='zsre')
    parser.add_argument('--model_path', type=str, default='../llama3.2-3b')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--span_len', type=int, default=10)
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
    print(len(original_corpus), len(continual_corpus))
    return original_corpus, continual_corpus


def select_tokens(continual_corpus, args):
    model_path = args.model_path
    model = 'llama'
    if 'qwen' in model_path:
        model = 'qwen'
    elif 'pythia' in model_path:
        model = 'pythia'
    top_k, span_len = 3, args.span_len
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_attentions=True)
    model.eval()
    model.to(f'cuda:{args.gpu}')
    
    input_ids_list = []
    for prompt in [sample['prompt'] for sample in continual_corpus]:
        input_ids = tokenizer(text=prompt, 
            padding=False, truncation=True, 
            add_special_tokens=True, return_token_type_ids=False,
            return_attention_mask=False)['input_ids']
        input_ids += [tokenizer.eos_token_id]
        input_ids_list.append(input_ids)
    
    all_results = []
    for ids in tqdm(input_ids_list, 'Find Important Snippet...'):
        input_ids = torch.tensor([ids])  # (1, seq_len)
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            attentions = outputs.attentions  # tuple(num_layers, batch, num_heads, seq_len, seq_len)
        attn = torch.stack(attentions, dim=0).squeeze(1)
        attn_mean = attn.mean(dim=(0,1))
        
        
        L = attn_mean.size(0)
        if L <= span_len:
            all_results.append(input_ids)
            continue
        
        
        col_sum = attn_mean.sum(dim=0)  # (L,)
        counts = torch.arange(1, L+1, device=attn_mean.device, dtype=attn_mean.dtype)
        token_scores = col_sum / counts
        snippet_scores = torch.zeros(L-span_len, device=token_scores.device)
        for i in range(span_len):
            snippet_scores += token_scores[i:L-span_len+i]

        selected = []
        used = torch.zeros_like(snippet_scores, dtype=torch.bool, device=snippet_scores.device)
        for _ in range(top_k):
            masked_scores = snippet_scores.clone()
            masked_scores[used] = -1  # 置极小值，保证不会选到

            idx = torch.argmax(masked_scores).item()
            if masked_scores[idx] < 0:
                break
            selected.append(idx)
            end = min(idx + span_len, L)
            used[idx:end] = True
        top_indices = sorted(selected)

        # 收集 spans
        spans = []
        for idx in top_indices:
            end = min(idx + span_len, input_ids.shape[1])
            span_ids = input_ids[0, idx:end]
            if len(span_ids) < span_len:
                continue
            spans.append(span_ids.tolist())

        all_results.append(spans)
    prompts = [tokenizer.decode(sub_tokens) for result in all_results for sub_tokens in result]
    with open(f'{args.dataset_name}/stoc_prompts-{args.model}.json', 'w') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)
    return prompts

def generate_replay_data(prompts, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.8, max_model_len=1024)
    sampling_params = SamplingParams(
        n = 3,
        temperature=1.0,      
        max_tokens=256, 
        repetition_penalty=1.05
    )
    results = llm.generate(prompts, sampling_params)
    
    texts = []
    token_ids = []
    for result in tqdm(results, desc='Collect Tokens'):
        prompt = result.prompt
        prompt_token_ids = result.prompt_token_ids
        for completion in result.outputs:
            generation_text = prompt + completion.text
            generation_token_ids = prompt_token_ids + list(completion.token_ids)
            texts.append(generation_text)
            token_ids.append(generation_token_ids[:256])
    with open(f'{args.dataset_name}/stoc_data-{args.model}.json', 'w') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
    with open(f'{args.dataset_name}/cpt_rehersal_stoc-{args.model}.pkl', "wb") as f:
        pickle.dump([{"input_ids": input_id} for input_id in token_ids], f)
    


if __name__ == '__main__':
    args = parse_args()
    original_corpus, continual_corpus = load_data(args.dataset_name)
    if os.path.exists(f'{args.dataset_name}/stoc_prompts-{args.model}.json'):
        with open(f'{args.dataset_name}/stoc_prompts-{args.model}.json', 'r') as f:
            prompts = json.load(f)
        print("Load Selected Tokens via attentiOn Contribution Finished.")
    else:
        prompts = select_tokens(continual_corpus, args)
        print("Selecting Tokens via attentiOn Contribution Finished.")
    if not os.path.exists(f'{args.dataset_name}/cpt_rehersal_stoc-{args.model}.pkl'):
        generate_replay_data(prompts, args)


