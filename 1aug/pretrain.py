import os, pathlib, shutil
import json, pickle
import argparse
import wandb
from tqdm import trange, tqdm
import torch

from transformers import Trainer, TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from dataset import PretrainDataset, FirstTokenAccuracyDataset


def parse_args():
    parser = argparse.ArgumentParser()
    
    # output and log
    parser.add_argument('--output_dir', type=str, default='outputs/pt-pythia/')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--wandb_use', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='PT')
    parser.add_argument('--wandb_name', type=str, default='5aug')
    parser.add_argument('--wandb_key', type=str, 
        default='')
    # data, tokenizer and model
    parser.add_argument('--data_path', type=str, default='data_pythia-160m/')
    parser.add_argument('--init_model_path', type=str, default='../pythia-160m',
        help='The path of the config, tokenizer.')
    # training args
    parser.add_argument('--optimizer', type=str, default='adamw_torch')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=80000)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--bf16', type=bool, default=True)
    args = parser.parse_args()
    return args

class FirstTokenAccuracyCallback_Train(TrainerCallback):
    def __init__(self,
                 first_token_accuracy_dataset: FirstTokenAccuracyDataset,
                 use_wandb: bool = True,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = first_token_accuracy_dataset
        self.attributes = self.dataset.attributes
        self.log_prefix = log_prefix
        
        self.use_wandb = use_wandb
        self.history = {}
        self.step_set = []
        self.calculation_strategy = calculation_strategy
        self.calculation_interval = calculation_interval

    def calculate_first_token_accuracy(self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs):
        model = kwargs['model']
        model = torch.nn.DataParallel(model)
        model.eval()
        attribute_to_count = {}
        for attribute in self.attributes:
            attribute_to_count[attribute] = {
                'total': 0,
                'hard_correct': 0,
                'soft_correct': 0
            }
        for i in trange(0, len(self.dataset), 
                args.train_batch_size  * 24, 
                desc='Calculating first token accuracy'):
            batch = self.dataset[i: i + args.train_batch_size]
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to('cuda')

            with torch.no_grad():
                logits = model(input_ids=inputs_ids).logits  # [bs, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()  # [bs, seq_len-1, vocab_size]
            shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))  # [bs * (seq_len-1), vocab_size]
            for attribute in self.attributes:
                index = torch.tensor([], dtype=torch.bool)
                first_token_list = []
                for item in batch['token_position']:
                    item_first_token_list = item[attribute]['first_token_list']
                    item_index = item[attribute]['index']
                    if len(item_first_token_list) > 0 and item_index[0] == torch.tensor(True):
                        item_first_token_list = item_first_token_list[1:]
                    first_token_list.extend(item_first_token_list)
                    item_index = item_index[1:]
                    index = torch.cat([index, item_index])
                first_token = torch.tensor(first_token_list, dtype=torch.int64)
                selected_logits = shift_logits[index]
                # hard correct
                prediction = torch.argmax(selected_logits, dim=-1)
                hard_correct = torch.sum(prediction.cpu() == first_token).item()
                # soft correct
                prediction_prob = torch.softmax(selected_logits, dim=-1)
                soft_correct = torch.sum(
                    torch.gather(prediction_prob.cpu(), dim=-1, index=first_token.unsqueeze(-1))).item()
                # record_result
                attribute_to_count[attribute]['total'] += first_token.shape[0]
                attribute_to_count[attribute]['hard_correct'] += hard_correct
                attribute_to_count[attribute]['soft_correct'] += soft_correct

        
        # log to wandb
        if self.use_wandb:
            for attribute, count in attribute_to_count.items():
                wandb.log({
                    f'{self.log_prefix}{attribute}_hardacc': count['hard_correct'] / count['total'],
                    f'{self.log_prefix}{attribute}_softacc': count['soft_correct'] / count['total'],
                })
        # record to history
        match self.calculation_strategy:
            case 'epoch':
                history_key = state.epoch
            case 'step' | 'end':
                history_key = state.global_step
            case _:
                raise ValueError(f'Invalid calculation strategy: {self.calculation_strategy}')
        self.history[history_key] = attribute_to_count
        output_path = os.path.join(args.output_dir, f'{self.log_prefix}first_token_accuracy_history.json')
        json.dump(self.history, open(output_path, 'w'), indent=4)

        model = model.module
        model.train()

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'step' and
            state.global_step % self.calculation_interval == 0):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'epoch' and
                int(state.epoch) % self.calculation_interval == 0):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_first_token_accuracy(args, state, control, **kwargs)
class FirstTokenAccuracyCallback_Test(TrainerCallback):
    def __init__(self,
                 first_token_accuracy_dataset: FirstTokenAccuracyDataset,
                 use_wandb: bool = True,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = first_token_accuracy_dataset
        self.attributes = self.dataset.attributes
        self.log_prefix = log_prefix
        self.n_gpu = torch.cuda.device_count()
        
        self.use_wandb = use_wandb
        self.history = {}
        self.step_set = []
        self.calculation_strategy = calculation_strategy
        self.calculation_interval = calculation_interval
    def calculate_first_token_accuracy(self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs):
        model = kwargs['model']
        model = torch.nn.DataParallel(model)
        model.eval()
        attribute_to_count = {}
        for attribute in self.attributes:
            attribute_to_count[attribute] = {
                'total': 0,
                'hard_correct': 0,
                'soft_correct': 0
            }
        for i in trange(0, len(self.dataset), 
                args.train_batch_size * self.n_gpu * 24, 
                desc='Calculating first token accuracy'):
            batch = self.dataset[i: i + args.train_batch_size]
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to('cuda')

            with torch.no_grad():
                logits = model(input_ids=inputs_ids).logits  # [bs, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()  # [bs, seq_len-1, vocab_size]
            shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))  # [bs * (seq_len-1), vocab_size]
            for attribute in self.attributes:
                index = torch.tensor([], dtype=torch.bool)
                first_token_list = []
                for item in batch['token_position']:
                    item_first_token_list = item[attribute]['first_token_list']
                    item_index = item[attribute]['index']
                    if len(item_first_token_list) > 0 and item_index[0] == torch.tensor(True):
                        item_first_token_list = item_first_token_list[1:]
                    first_token_list.extend(item_first_token_list)
                    item_index = item_index[1:]
                    index = torch.cat([index, item_index])
                first_token = torch.tensor(first_token_list, dtype=torch.int64)
                selected_logits = shift_logits[index]
                # hard correct
                prediction = torch.argmax(selected_logits, dim=-1)
                hard_correct = torch.sum(prediction.cpu() == first_token).item()
                # soft correct
                prediction_prob = torch.softmax(selected_logits, dim=-1)
                soft_correct = torch.sum(
                    torch.gather(prediction_prob.cpu(), dim=-1, index=first_token.unsqueeze(-1))).item()
                # record_result
                attribute_to_count[attribute]['total'] += first_token.shape[0]
                attribute_to_count[attribute]['hard_correct'] += hard_correct
                attribute_to_count[attribute]['soft_correct'] += soft_correct
        # log to wandb
        if self.use_wandb:
            for attribute, count in attribute_to_count.items():
                wandb.log({
                    f'{self.log_prefix}{attribute}_hardacc': count['hard_correct'] / count['total'],
                    f'{self.log_prefix}{attribute}_softacc': count['soft_correct'] / count['total'],
                })
        # record to history
        match self.calculation_strategy:
            case 'epoch':
                history_key = state.epoch
            case 'step' | 'end':
                history_key = state.global_step
            case _:
                raise ValueError(f'Invalid calculation strategy: {self.calculation_strategy}')
        self.history[history_key] = attribute_to_count
        output_path = os.path.join(args.output_dir, f'{self.log_prefix}first_token_accuracy_history.json')
        json.dump(self.history, open(output_path, 'w'), indent=4)

        model = model.module
        model.train()

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'step' and
            state.global_step % self.calculation_interval == 0):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == 'epoch' and
                int(state.epoch) % self.calculation_interval == 0):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_first_token_accuracy(args, state, control, **kwargs)
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
        remove_all_checkpoint: bool = False):
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
        save_strategy='steps', 
        save_steps=args.save_interval,  
        logging_steps=args.log_interval, 
        report_to = ["wandb"],
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
    model_config = AutoConfig.from_pretrained(
        args.init_model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.init_model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=True,
    )
    if 'qwen' in args.init_model_path:
        tokenizer.pad_token = tokenizer.eos_token
        print("qwen")
    elif 'pythia' in args.init_model_path:
        tokenizer.pad_token = tokenizer.unk_token
        print("pythia")
    model = AutoModelForCausalLM.from_config(model_config)
    print(model)
    
    # Set dataset
    train_dataset = PretrainDataset(
        data_path_list=[os.path.join(args.data_path, 'pt_train.pkl')],
        tokenizer=tokenizer)
    valid_dataset = FirstTokenAccuracyDataset(
        data_path_list=[os.path.join(args.data_path, 'pt_train.pkl')],
        tokenizer=tokenizer)
    test_dataset = FirstTokenAccuracyDataset(
        data_path_list=[os.path.join(args.data_path, 'pt_test.pkl')],
        tokenizer=tokenizer)
    
    # Set trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            FirstTokenAccuracyCallback_Train(valid_dataset, use_wandb=args.wandb_use,
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Train]'),
            FirstTokenAccuracyCallback_Test(test_dataset, use_wandb=args.wandb_use,
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Test]'),
            PreTrainingShuffleBiographyCallBack(),
        ],
    )
    train_and_save_model(trainer, training_args, remove_all_checkpoint=False)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)