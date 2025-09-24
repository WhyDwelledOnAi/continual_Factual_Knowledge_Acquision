export WANDB_BASE_URL=https://api.bandw.top


# python data_augmentation.py --dataset_name wiki_bio --gpu 1

# python process_data.py --dataset_name wiki_bio --model_path ../qwen2.5-0.5b

# Naive
CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/cpt-naive-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name naive-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# *****************************************************************

# Lamol-0.5
CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/cpt-lamol-0.5-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name lamol-0.5-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.5 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# Lamol-0.8
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/cpt-lamol-0.8-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name lamol-0.8-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.8 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# *****************************************************************
# STOC-0.5
CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/cpt-stoc-0.5-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name stoc-0.5-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.5 \
--freeze_layers -1 

# STOC-0.8
CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/cpt-stoc-0.8-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name stoc-0.8-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.8 \
--freeze_layers -1 

# *****************************************************************

# Freeze + Lamol-0.5
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/fcpt-lamol-0.5-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name freeze-lamol-0.5-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.5 \
--attn_data_replay_ratio 0.0 \
--freeze_layers 12

# Freeze + Lamol-0.8
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/fcpt-lamol-0.8-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name freeze-lamol-0.8-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.8 \
--attn_data_replay_ratio 0.0 \
--freeze_layers 12

# *****************************************************************

# Freeze + Stoc-0.5
CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/fcpt-stoc-0.5-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name freeze_stoc-0.5-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.5 \
--freeze_layers 12

# Freeze + Lamol-0.8
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../qwen2.5-0.5b \
--data_path wiki_bio/ \
--output_dir wiki_bio/outputs/fcpt-stoc-0.8-qwen/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Wiki_bio --wandb_name freeze_stoc-0.8-qwen \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.8 \
--freeze_layers 12

# *****************************************************************