export WANDB_BASE_URL=https://api.bandw.top


# python data_augmentation.py --dataset_name zsre --gpu 1

# python process_data.py --dataset_name zsre --model_path ../pythia-160m

# Naive
CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/cpt-naive-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name naive-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# *****************************************************************

# Lamol-0.5
CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/cpt-lamol-0.5-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name lamol-0.5-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.5 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# Lamol-0.8
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/cpt-lamol-0.8-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name lamol-0.8-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.8 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 

# *****************************************************************
# STOC-0.5
CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/cpt-stoc-0.5-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name stoc-0.5-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.5 \
--freeze_layers -1 

# STOC-0.8
CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/cpt-stoc-0.8-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name stoc-0.8-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.8 \
--freeze_layers -1 

# *****************************************************************

# Freeze + Lamol-0.5
CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/fcpt-lamol-0.5-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name freeze-lamol-0.5-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.5 \
--attn_data_replay_ratio 0.0 \
--freeze_layers 6 

# Freeze + Lamol-0.8
CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/fcpt-lamol-0.8-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name freeze-lamol-0.8-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.8 \
--attn_data_replay_ratio 0.0 \
--freeze_layers 6 

# *****************************************************************

# Freeze + Stoc-0.5
CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/fcpt-stoc-0.5-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name freeze_stoc-0.5-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.5 \
--freeze_layers 6 

# Freeze + Lamol-0.8
CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
--init_model_path ../pythia-160m \
--data_path zsre/ \
--output_dir zsre/outputs/fcpt-stoc-0.8-pythia/ \
--save_interval 8000 --log_interval 10 --eval_interval 500 \
--wandb_project Zsre --wandb_name freeze_stoc-0.8-pythia \
--warmup_steps 500 --max_steps 8000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 48 --accumulate_steps 1 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.8 \
--freeze_layers 6 

# *****************************************************************