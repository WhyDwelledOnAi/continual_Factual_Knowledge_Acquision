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
from transformers import GPTNeoXForCausalLM, Qwen2ForCausalLM

from dataset import PretrainDataset, FirstTokenAccuracyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # output and log
    parser.add_argument('--output_dir', type=str, default='outputs/continue_pretrain/')
    parser.add_argument('--save_interval', type=int, default=500)
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
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--bf16', type=bool, default=True)

    # [Some Continual Learning Baselines]
    # The ratio is the percentage of the cpt_train data
    
    # Cheating methods
    parser.add_argument('--all_data_replay_ratio', type=float, default=0.)
    parser.add_argument('--half_data_replay_ratio', type=float, default=0.)
    # Lamol: Language modeling for lifelong language learning.
    parser.add_argument('--naive_data_replay_ratio', type=float, default=0.)
    # Ours
    parser.add_argument('--attn_data_replay_ratio', type=float, default=0.)
    # Spurious Forgetting in Continual Learning of Language Models.
    parser.add_argument('--freeze_layers', type=int, default=-1) # -1 means not freezing
    # Overcoming catastrophic forgetting in neural networks.
    parser.add_argument('--regularization', type=float, default=0.0)
    parser.add_argument('--fisher_sample_size', type=int, default=32)
    # Gradient projection memory for continual learning.
    parser.add_argument('--gpm_threshold', type=float, default=0.0) 
    parser.add_argument('--gpm_sample_size', type=int, default=32)
    
    args = parser.parse_args()
    return args



class MyTrainer(Trainer):
    def __init__(self, *args,  **kwargs):
        self.regularization = kwargs.pop('regularization', 0.0)
        self.fisher_dataset = kwargs.pop('fisher_dataset', None)
        self.fisher_max_batches = kwargs.pop('fisher_sample_size', 
                len(self.fisher_dataset) if self.fisher_dataset is not None else 0)
        
        self.freeze_layers = kwargs.pop('freeze_layers', -1)
        
        
        
        self.gpm_threshold = kwargs.pop('gpm_threshold', 0.0)
        self.gpm_dataset = kwargs.pop('gpm_dataset', None)
        self.gpm_calib_batches = kwargs.pop('gpm_sample_size', 
                len(self.gpm_dataset) if self.gpm_dataset is not None else 0)


        super().__init__(*args, **kwargs)

        # Set reference model for EWC regularization 
        self.reference_model = None
        if (self.regularization > 0.0 and self.fisher_dataset is not None) or \
           (self.gpm_threshold > 0.0 and self.gpm_calib_batches > 0):
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # Calculate Fisher Matrix
        self.fisher = None
        if self.regularization > 0.0 and self.fisher_dataset is not None:
            self.fisher = self._fisher_matrix_diag(self.fisher_dataset)
            print("Fisher Information Matrix Calculated.")

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
            print(f"The first {self.freeze_layers} layers are frozen.")
        
        # Set Gradient Projection Memory
        self.gpm_param_to_proj = {}
        if self.gpm_threshold > 0.:
            self.set_gradient_basis()
            print("Gradient Projection Memory is set.")
    

    def _fisher_matrix_diag(self, dataset: PretrainDataset):
        model = self.model
        model.train()
        device = next(model.parameters()).device
        fisher = {}
        for name, p in model.named_parameters():
            fisher[name] = torch.zeros_like(p, device=device)
        
        batch_size = self.args.per_device_train_batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        for batch_idx, batch in tqdm(enumerate(data_loader), desc='Calculating Fisher Matrix'):
            if self.fisher_max_batches and batch_idx >= self.fisher_max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()

            bsz = input_ids.shape[0]
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += bsz * (p.grad.detach() ** 2)
        
        total_count = dataset.__len__()
        for name in fisher.keys():
            fisher[name] = (fisher[name] / total_count).detach()
            fisher[name] = torch.autograd.Variable(fisher[name], requires_grad=False)
        return fisher
    def _ewc_regularization(self, current_model: torch.nn.Module) -> torch.Tensor:
        if self.fisher is None:
            return torch.tensor(0.0, device=next(current_model.parameters()).device)
        reg_loss = 0.0
        ref_params = {name: p for name, p in self.reference_model.named_parameters()}
        for name, param in current_model.named_parameters():
            if (name in ref_params) and (name in self.fisher):
                ref_param = ref_params[name].to(param.device)
                fisher_diag = self.fisher[name].to(param.device)
                param_diff = param - ref_param
                reg_loss += torch.sum(fisher_diag * param_diff.pow(2))
        return reg_loss / 2.0
    # Add EWC Regularization Term to Loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            total_loss, outputs = super().compute_loss(model, inputs, 
                return_outputs, num_items_in_batch)
        else:
            total_loss = super().compute_loss(model, inputs, 
                return_outputs, num_items_in_batch)
        # EWC
        if self.fisher is not None:
            current_model = model.module if hasattr(model, "module") else model
            ewc_loss = self._ewc_regularization(current_model)
            total_loss += self.regularization * ewc_loss

        return (total_loss, outputs) if return_outputs else total_loss
    

    def _iter_target_linear_modules(self, model):
        modules = []
        for li, layer in enumerate(model.gpt_neox.layers):
            attn = getattr(layer, 'attention', None)
            mlp = getattr(layer, 'mlp', None)
            if attn is not None:
                if hasattr(attn, 'query_key_value'):
                    modules.append((f'gpt_neox.layers.{li}.attention.query_key_value', attn.query_key_value))
                if hasattr(attn, 'dense'):
                    modules.append((f'gpt_neox.layers.{li}.attention.dense', attn.dense))
            if mlp is not None:
                if hasattr(mlp, 'dense_h_to_4h'):
                    modules.append((f'gpt_neox.layers.{li}.mlp.dense_h_to_4h', mlp.dense_h_to_4h))
                if hasattr(mlp, 'dense_4h_to_h'):
                    modules.append((f'gpt_neox.layers.{li}.mlp.dense_4h_to_h', mlp.dense_4h_to_h))
        return modules
    def set_gradient_basis(self):
        if self.reference_model is None or self.gpm_dataset is None:
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reference_model.to(device)

        self.gpm_target_linear_paths = [name for name, _ in \
            self._iter_target_linear_modules(self.reference_model)]
        activation_buffers = {name: [] for name in self.gpm_target_linear_paths}

        
        hooks = []
        module_dict = dict(self._iter_target_linear_modules(self.reference_model))
        for name, module in module_dict.items():
            def make_hook(key):
                def hook(module, inputs, output):
                    inp = inputs[0]
                    if inp is None:
                        return
                    if inp.dim() == 3:
                        N = inp.size(0) * inp.size(1)
                        feat = inp.reshape(N, inp.size(-1))
                    else:
                        feat = inp.reshape(-1, inp.size(-1))
                    activation_buffers[key].append(feat.detach().to('cpu', dtype=torch.float32))
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))
        calib_bs = self.args.per_device_train_batch_size
        dl = DataLoader(self.gpm_dataset, batch_size=calib_bs, shuffle=True)
        self.reference_model.eval()
        with torch.no_grad():
            for bi, batch in tqdm(enumerate(dl), desc='Collecting Activation for GPM'):
                if bi >= self.gpm_calib_batches:
                    break
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                self.reference_model(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()

        param_name_prefix = ''
        for n, _ in self.model.named_parameters():
            if n.startswith('module.'):
                param_name_prefix = 'module.'
                break

        for layer_path, feats in tqdm(activation_buffers.items(), desc='Building GPM'):
            if len(feats) == 0:
                continue
            X = torch.cat(feats, dim=0)  # [N, H]
            X = X.t().contiguous()  # [H, N]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            S2 = (S ** 2)
            sval_total = torch.sum(S2) + 1e-12
            cumsum = torch.cumsum(S2 / sval_total, dim=0)
            r = int(torch.sum(cumsum < self.gpm_threshold).item())
            if r <= 0:
                continue
            B = U[:, :r]  # [H, r]
            P = (B @ B.t()).to(device)  # [H, H]
            param_name = f'{param_name_prefix}{layer_path}.weight'
            self.gpm_param_to_proj[param_name] = P

    # Add Gradient Projection to Training Step
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean() 

        loss *= self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if len(self.gpm_param_to_proj) > 0:
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                
                proj = self.gpm_param_to_proj.get(name, None)
                if proj is None and name.startswith('module.'):
                    proj = self.gpm_param_to_proj.get(name[len('module.'):], None)
                if proj is None:
                    continue
                if param.grad.dim() == 2:
                    # grad: [out, in], proj: [in, in]
                    param.grad.data = param.grad.data - torch.matmul(param.grad.data, proj.to(param.grad.device))

        return loss.detach() / self.args.gradient_accumulation_steps


class FirstTokenAccuracyCallback_PretrainKnowledge(TrainerCallback):
    """
        Used for evaluation by predicting first token.
    """
    def __init__(self,
                 first_token_accuracy_dataset: FirstTokenAccuracyDataset,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = first_token_accuracy_dataset
        self.attributes = self.dataset.attributes
        self.log_prefix = log_prefix
        
        
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
        
        model.eval()
        attribute_to_count = {}
        for attribute in self.attributes:
            attribute_to_count[attribute] = {
                'total': 0,
                'hard_correct': 0,
                'soft_correct': 0
            }
        for i in trange(0, len(self.dataset), 
                args.train_batch_size * 24, 
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
class FirstTokenAccuracyCallback_ContinualKnowledge(TrainerCallback):
    """
        Used for evaluation by predicting first token.
    """
    def __init__(self,
                 first_token_accuracy_dataset: FirstTokenAccuracyDataset,
                 calculation_strategy: str = 'step',
                 calculation_interval: int = 100,
                 log_prefix: str = ''):
        assert calculation_strategy in ['epoch', 'step', 'end']
        self.dataset = first_token_accuracy_dataset
        self.attributes = self.dataset.attributes
        self.log_prefix = log_prefix
        
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
        model.eval()
        attribute_to_count = {}
        for attribute in self.attributes:
            attribute_to_count[attribute] = {
                'total': 0,
                'hard_correct': 0,
                'soft_correct': 0
            }
        for i in trange(0, len(self.dataset), 
                args.train_batch_size* 24, 
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
        model_max_length=512,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if 'qwen' in args.init_model_path:
        tokenizer.pad_token = tokenizer.eos_token
        print("qwen")
    elif 'pythia' in args.init_model_path:
        tokenizer.pad_token = tokenizer.unk_token
        print("pythia")
    model = AutoModelForCausalLM.from_pretrained(args.init_model_path)
    print(model)
    
    
    # Set dataset
    train_data_path_list = []
    if args.all_data_replay_ratio > 0.:
        assert os.path.exists(os.path.join(args.data_path, 'cpt_rehersal_all.pkl')), \
            "cpt_rehersal_all.pkl is not found."
        train_data_path_list.append(os.path.join(args.data_path, 'cpt_rehersal_all.pkl'))
        cpt_train_times = int(args.all_data_replay_ratio / (1-args.all_data_replay_ratio))
        train_data_path_list.extend([os.path.join(args.data_path, 'cpt_train.pkl'),]*cpt_train_times)
        print("Using cpt_rehersal_all.pkl for Rehersal.")
        print(f"cpt_train.pkl * {cpt_train_times}")
    elif args.half_data_replay_ratio > 0.:
        assert os.path.exists(os.path.join(args.data_path, 'cpt_rehersal_half.pkl')), \
            "cpt_rehersal_half.pkl is not found."
        train_data_path_list.append(os.path.join(args.data_path, 'cpt_rehersal_half.pkl'))
        cpt_train_times = int(args.half_data_replay_ratio / (1-args.half_data_replay_ratio))
        train_data_path_list.extend([os.path.join(args.data_path, 'cpt_train.pkl'),]*cpt_train_times)
        print("Using cpt_rehersal_half.pkl for Rehersal.")
        print(f"cpt_train.pkl * {cpt_train_times}")
    elif args.naive_data_replay_ratio > 0.:
        assert os.path.exists(os.path.join(args.data_path, 'cpt_rehersal_naive.pkl')), \
            "cpt_rehersal_naive.pkl is not found."
        train_data_path_list.append(os.path.join(args.data_path, 'cpt_rehersal_naive.pkl'))
        cpt_train_times = int(args.naive_data_replay_ratio / (1-args.naive_data_replay_ratio))
        train_data_path_list.extend([os.path.join(args.data_path, 'cpt_train.pkl'),]*cpt_train_times)
        print("Using cpt_rehersal_naive.pkl for Rehersal.")
        print(f"cpt_train.pkl * {cpt_train_times}")
    elif args.attn_data_replay_ratio > 0.:
        assert os.path.exists(os.path.join(args.data_path, 'cpt_rehersal_attn.pkl')), \
            "cpt_rehersal_attn.pkl is not found."
        train_data_path_list.append(os.path.join(args.data_path, 'cpt_rehersal_attn.pkl'))
        cpt_train_times = int(args.attn_data_replay_ratio / (1-args.attn_data_replay_ratio))
        train_data_path_list.extend([os.path.join(args.data_path, 'cpt_train.pkl'),]*cpt_train_times)
        print("Using cpt_rehersal_attn.pkl for Rehersal.")
        print(f"cpt_train.pkl * {cpt_train_times}")
    else:
        train_data_path_list = [os.path.join(args.data_path, 'cpt_train.pkl'),]
        print("No Rehersal.")

    train_dataset = PretrainDataset(
        data_path_list = train_data_path_list,
        tokenizer=tokenizer)
    if args.regularization > 0.0:
        fisher_dataset = PretrainDataset(
            data_path_list = [os.path.join(args.data_path, 'pt_train.pkl'),],
            tokenizer=tokenizer)
    else:
        fisher_dataset = None
    if args.gpm_threshold > 0.0:
        gpm_dataset = PretrainDataset(
            data_path_list = [os.path.join(args.data_path, 'pt_train.pkl'),],
            tokenizer=tokenizer)
    else:
        gpm_dataset = None
    original_dataset = FirstTokenAccuracyDataset(
        data_path_list=[os.path.join(args.data_path, 'pt_test.pkl')],
        tokenizer=tokenizer)
    continual_dataset = FirstTokenAccuracyDataset(
        data_path_list=[os.path.join(args.data_path, 'cpt_test.pkl')],
        tokenizer=tokenizer)

    
    # Set trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            FirstTokenAccuracyCallback_PretrainKnowledge(original_dataset, 
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Original]'),
            FirstTokenAccuracyCallback_ContinualKnowledge(continual_dataset, 
                calculation_strategy='step', calculation_interval=args.eval_interval,
                log_prefix='[Continual]'),
            PreTrainingShuffleBiographyCallBack(),
        ],
        # Hyper-Parameters of Some Continual Learning Baselines
        regularization=args.regularization,
        fisher_dataset=fisher_dataset,
        fisher_sample_size=args.fisher_sample_size,
        
        freeze_layers=args.freeze_layers,
        
        gpm_threshold=args.gpm_threshold,
        gpm_dataset=gpm_dataset,
        gpm_sample_size=args.gpm_sample_size,

    )
    train_and_save_model(trainer, training_args)




if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)