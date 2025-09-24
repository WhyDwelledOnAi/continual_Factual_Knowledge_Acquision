import torch
import pickle, os, random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model = 'qwen'
model_path = f"outputs/pt-{model}/final_model"
if model == 'pythia':
    output_root = 'data_pythia-160m'
elif model == 'qwen':
    output_root = 'data_qwen2.5-0.5b'

with open(f"{output_root}/cpt_train.pkl", "rb") as f:
    data = pickle.load(f)
input_ids = [item['input_ids'] for item in data]
target_num_tokens = sum(len(ids) for ids in input_ids)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, output_attentions=True)
model.eval()

def process_input_ids(input_ids_list, top_k=1, span_len=4):
    all_results = []
    for ids in tqdm(input_ids_list, 'Find Important Spans'):
        input_ids = torch.tensor([ids])  # (1, seq_len)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            attentions = outputs.attentions  # tuple(num_layers, batch, num_heads, seq_len, seq_len)
        attn = torch.stack(attentions, dim=0).squeeze(1)
        attn_mean = attn.mean(dim=(0,1))
        
        L = attn_mean.size(0)
        col_sum = attn_mean.sum(dim=0)  # (L,)
        counts = torch.arange(1, L+1, device=attn_mean.device, dtype=attn_mean.dtype)
        token_scores = col_sum / counts

        # 每个 token 的 attention score
        selected = []
        used = torch.zeros(L, dtype=torch.bool, device=attn_mean.device)
        for _ in range(top_k):
            masked_scores = token_scores.clone()
            masked_scores[used] = -1 

            idx = torch.argmax(masked_scores).item()
            if masked_scores[idx] < 0:
                break
            selected.append(idx)
            end = min(idx + span_len, L)
            used[idx:end] = True
        top_indices = sorted(selected)

        spans = []
        for idx in top_indices:
            end = min(idx + span_len, input_ids.shape[1])
            span_ids = input_ids[0, idx:end]
            if len(span_ids) < span_len:
                continue
            spans.append(span_ids.tolist())

        all_results.append(spans)
    return all_results
if os.path.exists(f'{output_root}/cpt_rehersal_attn_prompt.pkl'):
    with open(f'{output_root}/cpt_rehersal_attn_prompt.pkl', 'rb') as f:
        prompts = pickle.load(f)
    print("Load Selected Tokens via attentiOn Contribution Finished.")
else:
    with open(f"{output_root}/cpt_train.pkl", "rb") as f:
        data = pickle.load(f)[:10000]
    input_ids_list = [data_item['input_ids'] for data_item in data]
    all_results = process_input_ids(input_ids_list, top_k=3, span_len=3)
    prompts = [tokenizer.decode(sub_tokens) for result in all_results for sub_tokens in result]
    with open(f'{output_root}/cpt_rehersal_attn_prompt.pkl', 'wb') as f:
        pickle.dump(prompts, f)
    print("Selecting Tokens via attentiOn Contribution Finished.")

llm = LLM(model=model_path, gpu_memory_utilization=0.8)
sampling_params = SamplingParams(
    n = 3,
    temperature=1.0,      
    max_tokens=512,       
    stop=None,            
    ignore_eos=True,      
    repetition_penalty=1.05
)
results = llm.generate(prompts, sampling_params)
token_ids = [list(completion.token_ids) for result in results for completion in result.outputs]
token_ids = []
for result in tqdm(results, desc='Collect Tokens'):
    prompt_token_ids = result.prompt_token_ids
    for completion in result.outputs:
        generation_token_ids = prompt_token_ids + list(completion.token_ids)
        token_ids.append(generation_token_ids[:512])


input_ids = random.choices(token_ids, k=min(target_num_tokens//512, len(token_ids)))
num_tokens = sum(len(ids) for ids in input_ids)
print("target token num:", target_num_tokens) # 15299555 
print("generated token num:", num_tokens)     # 15299072 

with open(f'{output_root}/cpt_rehersal_attn.pkl', "wb") as f:
    pickle.dump([{"input_ids": input_id} for input_id in input_ids], f)


