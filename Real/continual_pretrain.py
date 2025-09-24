import os, pathlib, shutil, copy
import json, pickle
import argparse
import wandb
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from dataset import CPTDataset, FluencyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # output and log
    parser.add_argument('--output_dir', type=str, default='outputs/continue_pretrain/')
    parser.add_argument('--save_interval', type=int, default=4000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--wandb_project', type=str, default='continual_pretrain')
    parser.add_argument('--wandb_name', type=str, default='default_name')
    parser.add_argument('--wandb_key', type=str, 
        default='')
    
    # data, tokenizer and model
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--init_model_path', type=str, 
        default='outputs/pretrain/5aug-final_model',
        help='The path of the config, tokenizer.')
    
    # training args
    parser.add_argument('--optimizer', type=str, default='adamw_torch')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--lr_scheduler_type', type=str,
        default='cosine_with_min_lr')
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=8000)
    parser.add_argument('--max_lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--bf16', type=bool, default=True)

    # [Some Continual Learning Baselines]
    # The ratio is the percentage of the cpt_train data
    
    # Lamol: Language modeling for lifelong language learning.
    parser.add_argument('--naive_data_replay_ratio', type=float, default=0.)
    # Ours
    parser.add_argument('--attn_data_replay_ratio', type=float, default=0.)
    # Spurious Forgetting in Continual Learning of Language Models.
    parser.add_argument('--freeze_layers', type=int, default=-1) # -1 means not freezing
    
    args = parser.parse_args()
    args.model = 'llama'
    if 'qwen' in args.init_model_path:
        args.model = 'qwen'
    elif 'pythia' in args.init_model_path:
        args.model = 'pythia'
    return args



class MyTrainer(Trainer):
    def __init__(self, *args,  **kwargs):
        self.freeze_layers = kwargs.pop('freeze_layers', -1)
        super().__init__(*args, **kwargs)

        # Set Freezing Layers
        if self.freeze_layers > 0:
            if type(self.model) == GPTNeoXForCausalLM:
                for param in self.model.gpt_neox.embed_in.parameters():
                    param.requires_grad = False
                for i in range(self.freeze_layers):
                    for param in self.model.gpt_neox.layers[i].parameters():
                        param.requires_grad = False
            elif type(self.model) == Qwen2ForCausalLM:
                for param in self.model.model.embed_tokens.parameters():
                    param.requires_grad = False
                for i in range(self.freeze_layers):
                    for param in self.model.model.layers[i].parameters():
                        param.requires_grad = False
            elif type(self.model) == LlamaForCausalLM:
                for param in self.model.model.embed_tokens.parameters():
                    param.requires_grad = False
                for i in range(self.freeze_layers):
                    for param in self.model.model.layers[i].parameters():
                        param.requires_grad = False
            print(f"The first {self.freeze_layers} layers are frozen.")
    


class FluencyCallback_PretrainKnowledge(TrainerCallback): 
    def __init__(self,
                 fluency_dataset: FluencyDataset,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = fluency_dataset
        self.log_prefix = log_prefix
        
        self.calculation_strategy = calculation_strategy
        self.calculation_interval = calculation_interval

    def calculate_fluency(self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs):
        model = kwargs['model']
        model.eval()
        
        total_prob = 0.0
        total_tokens = 0
        for i in trange(0, len(self.dataset), 
                args.train_batch_size*24, 
                desc='Calculating first token accuracy'):
            batch = self.dataset[i: i + args.train_batch_size]
            
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to('cuda')
            labels_ids = torch.cat([ids.unsqueeze(0) for ids in batch['label_ids']], dim=0)
            labels_ids = labels_ids.to('cuda')

            with torch.no_grad():
                logits = model(input_ids=inputs_ids).logits  # [bs, seq_len, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous() # [bs, seq_len-1, vocab_size]
            shift_labels = labels_ids[:, 1:]  # [bs, seq_len-1]
            
            shift_labels_copy = shift_labels.clone()
            shift_labels_copy[shift_labels_copy == IGNORE_TOKEN_ID] = 0 
            
            
            probs = torch.nn.functional.softmax(shift_logits, dim=-1)  # [bs, seq_len, vocab_size]
            target_probs = probs.gather(-1, shift_labels_copy.unsqueeze(-1)).squeeze(-1)  # [bs, seq_len]

            
            mask = shift_labels != IGNORE_TOKEN_ID
            total_prob += (target_probs * mask).sum().item()
            total_tokens += mask.sum().item()
            
        soft_accuracy = total_prob / total_tokens if total_tokens > 0 else 0.0
                
        # log to wandb
        wandb.log({
            f'{self.log_prefix}Fluency': soft_accuracy,
        })
        
        model.train()

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_fluency(args, state, control, **kwargs)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'step' and
            state.global_step % self.calculation_interval == 0):
            self.calculate_fluency(args, state, control, **kwargs)
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'epoch' and
                int(state.epoch) % self.calculation_interval == 0):
            self.calculate_fluency(args, state, control, **kwargs)
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_fluency(args, state, control, **kwargs)
class FluencyCallback_ContinualKnowledge(TrainerCallback): 
    def __init__(self,
                 fluency_dataset: FluencyDataset,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = fluency_dataset
        self.log_prefix = log_prefix
        
        self.calculation_strategy = calculation_strategy
        self.calculation_interval = calculation_interval

    def calculate_fluency(self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs):
        model = kwargs['model']
        model.eval()
        
        total_prob = 0.0
        total_tokens = 0
        for i in trange(0, len(self.dataset), 
                args.train_batch_size * 24, 
                desc='Calculating first token accuracy'):
            batch = self.dataset[i: i + args.train_batch_size]
            
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to('cuda')
            labels_ids = torch.cat([ids.unsqueeze(0) for ids in batch['label_ids']], dim=0)
            labels_ids = labels_ids.to('cuda')

            with torch.no_grad():
                logits = model(input_ids=inputs_ids).logits  # [bs, seq_len, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous() # [bs, seq_len-1, vocab_size]
            shift_labels = labels_ids[:, 1:]  # [bs, seq_len-1]
            
            shift_labels_copy = shift_labels.clone()
            shift_labels_copy[shift_labels_copy == IGNORE_TOKEN_ID] = 0 
            
            
            probs = torch.nn.functional.softmax(shift_logits, dim=-1)  # [bs, seq_len, vocab_size]
            target_probs = probs.gather(-1, shift_labels_copy.unsqueeze(-1)).squeeze(-1)  # [bs, seq_len]

            
            mask = shift_labels != IGNORE_TOKEN_ID
            total_prob += (target_probs * mask).sum().item()
            total_tokens += mask.sum().item()
            
        soft_accuracy = total_prob / total_tokens if total_tokens > 0 else 0.0
                
        # log to wandb
        wandb.log({
            f'{self.log_prefix}Fluency': soft_accuracy,
        })
        
        model.train()

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_fluency(args, state, control, **kwargs)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'step' and
            state.global_step % self.calculation_interval == 0):
            self.calculate_fluency(args, state, control, **kwargs)
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'epoch' and
                int(state.epoch) % self.calculation_interval == 0):
            self.calculate_fluency(args, state, control, **kwargs)
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_fluency(args, state, control, **kwargs)
class PreTrainingShuffleBiographyCallBack(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs['train_dataloader'].dataset.construct_dataset()
        

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    trainer._save(output_dir, state_dict=cpu_state_dict)
def train_and_save_model(
        trainer: Trainer, 
        training_args: TrainingArguments, 
        remove_all_checkpoint: bool = True):
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    final_model_path = os.path.join(training_args.output_dir, 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_path)
    if remove_all_checkpoint:
        for checkpoint_dir in pathlib.Path(training_args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_dir)

def main(args):
    # Set configurations
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        # self-defined evaluation in the callback
        eval_strategy='no', 
        # output and logging
        output_dir=args.output_dir,
        save_strategy='no', 
        save_steps=args.save_interval,  
        logging_steps=args.log_interval, 
        report_to=['wandb'],
        # optimizer
        optim=args.optimizer,  
        weight_decay=args.weight_decay, 
        adam_epsilon=args.adam_epsilon,  
        # batch and multi-gpu
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate_steps,
        # lr and scheduler
        learning_rate=args.max_lr,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        warmup_steps= args.warmup_steps, 
        max_steps = args.max_steps,
        bf16=args.bf16,)
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=args.__dict__)
    
    # Set tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.init_model_path,
        model_max_length=256,
        padding_side="right",
        use_fast=True,
    )
    if 'qwen' in args.init_model_path:
        tokenizer.pad_token = tokenizer.eos_token
        print("qwen")
    elif 'pythia' in args.init_model_path:
        tokenizer.pad_token = tokenizer.unk_token
        print("pythia")
    elif 'llama' in args.init_model_path:
        tokenizer.pad_token = tokenizer.eos_token
        print("llama")
    
    model = AutoModelForCausalLM.from_pretrained(args.init_model_path)
    print(model)
    
    
    # Set dataset
    train_data_path_list, train_data_ratio_list = [], []
    if args.naive_data_replay_ratio > 0.:
        replay_data_path = os.path.join(args.data_path, f'cpt_rehersal_lamol-{args.model}.pkl')
        assert os.path.exists(replay_data_path), f"{replay_data_path} is not found."
        
        train_data_path_list.append(replay_data_path)
        train_data_ratio_list.append(1-args.naive_data_replay_ratio)
        train_data_path_list.append(os.path.join(args.data_path, f'cpt_train-{args.model}.pkl'))
        train_data_ratio_list.append(1-args.naive_data_replay_ratio)
        print(f"Using {replay_data_path} for Rehersal.")
        print(f"Mixing ratio: {args.naive_data_replay_ratio}")
    elif args.attn_data_replay_ratio > 0.:
        replay_data_path = os.path.join(args.data_path, f'cpt_rehersal_stoc-{args.model}.pkl')
        assert os.path.exists(replay_data_path), f"{replay_data_path} is not found."
        
        train_data_path_list.append(replay_data_path)
        train_data_ratio_list.append(1-args.attn_data_replay_ratio)
        train_data_path_list.append(os.path.join(args.data_path, f'cpt_train-{args.model}.pkl'))
        train_data_ratio_list.append(1-args.attn_data_replay_ratio)
        print(f"Using {replay_data_path} for Rehersal.")
        print(f"Mixing ratio: {args.attn_data_replay_ratio}")
    else:
        train_data_path_list = [os.path.join(args.data_path, f'cpt_train-{args.model}.pkl'),]
        train_data_ratio_list = [1.0]
        print("No Rehersal.")

    train_dataset = CPTDataset(
        data_path_list = train_data_path_list,
        data_ratio_list= train_data_ratio_list,
        tokenizer=tokenizer)
    original_dataset = FluencyDataset(
        data_path_list=[os.path.join(args.data_path, f'pt_test-{args.model}.pkl')],
        tokenizer=tokenizer)
    continual_dataset = FluencyDataset(
        data_path_list=[os.path.join(args.data_path, f'cpt_test-{args.model}.pkl')],
        tokenizer=tokenizer)

    
    # Set trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            FluencyCallback_PretrainKnowledge(original_dataset, 
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Original]'),
            FluencyCallback_ContinualKnowledge(continual_dataset, 
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Continual]'),
            PreTrainingShuffleBiographyCallBack(),
        ],
        # Hyper-Parameters of Some Continual Learning Baselines
        freeze_layers=args.freeze_layers,

    )
    train_and_save_model(trainer, training_args)




if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)