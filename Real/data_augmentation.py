
from tqdm import tqdm, trange
import json, pickle
import argparse, os



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default='zsre')
    parser.add_argument('--tokenizer_path', type=str, default='../llama3.2-3b')
    parser.add_argument('--rewrite_llm', type=str, default='../llama3.1-8b-instruct')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
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

def rewrite_continual_samples(dataset_name, model_path, continual_corpus, gpu=0):
    from vllm import LLM, SamplingParams
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    
    llm = LLM(model=model_path, gpu_memory_utilization=0.8, max_model_len=1024)
    sampling_params = SamplingParams(
        n = 5,
        temperature=0.6,      
        max_tokens=256,       
        repetition_penalty=1.05
    )
    
    query_template = (
        "Please Re-write the following statement directly to generate one new text. Don't do anything more.\n"
        "Statement: {}\n"
        "Rewritten text:"
        )
    queries = []
    for example in continual_corpus:
        prompt = example['prompt']+' '
        for truth in example['ground_truth']:
            statement = prompt + truth + '.'
            queries.append(query_template.format(statement))
    results = llm.generate(queries, sampling_params)
    
    augmented_corpus = []
    for i in range(len(continual_corpus)):
        example = continual_corpus[i]
        responses = results[i].outputs
        for response in responses:
            augmented_prompt = response.text
            augmented_corpus.append(
                {'prompt': augmented_prompt, 'ground_truth':example['ground_truth']}
            )
    with open(f'{dataset_name}/augmented_continual.json', 'w') as f:
        json.dump(augmented_corpus, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parse_args()
    original_corpus, continual_corpus = load_data(args.dataset_name)
    rewrite_continual_samples(args.dataset_name, args.rewrite_llm, continual_corpus, args.gpu)