export WANDB_BASE_URL=https://api.bandw.top

# python synthesis_data.py --tokenizer qwen2.5-0.5b
# *******************************************************

# Pretraining
# CUDA_VISIBLE_DEVICES=0 python pretrain.py \
# --output_dir outputs/pt-qwen/ \
# --save_interval 80000 \
# --log_interval 10 \
# --eval_interval 20000 \
# --wandb_project PT \
# --wandb_name '5aug-qwen' \
# --data_path data_qwen2.5-0.5b/ \
# --init_model_path ../qwen2.5-0.5b \
# --warmup_steps 4000 \
# --max_steps 320000 \
# --batch_size 12 \
# --accumulate_steps 4 \
# --min_lr 5e-5

# python generate_answer_pt.py --gpu 0 \
# --model qwen2.5-0.5b --mode train \
# --model_path outputs/pt-qwen/final_model 

# python generate_answer_pt.py --gpu 0 \
# --model qwen2.5-0.5b --mode test \
# --model_path outputs/pt-qwen/final_model 

# *******************************************************

# Scratch Continual Pretraining
CUDA_VISIBLE_DEVICES=1 python cpt_zero.py \
--init_model_path outputs/pt-qwen/final_model \
--data_path data_qwen2.5-0.5b/ \
--output_dir outputs/cpt-scratch-qwen/ \
--save_interval 20000 --log_interval 10 --eval_interval 5000 \
--wandb_project CPT2 --wandb_name Scratch-qwen \
--warmup_steps 1000 --max_steps 80000 \
--max_lr 5e-5 --min_lr 1e-5 \
--batch_size 12 --accumulate_steps 4 \
--all_data_replay_ratio 0.0 \
--half_data_replay_ratio 0.0 \
--naive_data_replay_ratio 0.0 \
--attn_data_replay_ratio 0.0 \
--freeze_layers -1 \
--regularization 0.0 --fisher_sample_size 256 \
--gpm_threshold -1 --gpm_sample_size 256


# *******************************************************

# Naive Continual Pretraining
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-naive-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT --wandb_name Naive-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-naive-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-naive-qwen/final_model 

# *******************************************************


# 0.9 All Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-all-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name All-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.9 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-all-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-all-0.9-qwen/final_model 

# *******************************************************

# 0.8 All Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-all-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name All-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.8 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-all-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-all-0.8-qwen/final_model 

# *******************************************************


# 0.67 All Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-all-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name All-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.67 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-all-0.67-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-all-0.67-qwen/final_model 

# *******************************************************


# 0.9 Half Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-half-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Half-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.9 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-half-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-half-0.9-qwen/final_model 

# *******************************************************


# 0.8 Half Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-half-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Half-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.8 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-half-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-half-0.8-qwen/final_model 

# *******************************************************


# 0.67 Half Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-half-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Half-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.67 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-half-0.67-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-half-0.67-qwen/final_model 

# *******************************************************

# 0.9 Lamol Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-lamol-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Lamol-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.9 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-lamol-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-lamol-0.9-qwen/final_model 

# *******************************************************

# 0.8 Lamol Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=2 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-lamol-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Lamol-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.8 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-lamol-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-lamol-0.8-qwen/final_model 

# *******************************************************

# 0.67 Lamol Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-lamol-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Lamol-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.67 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-lamol-0.67-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-lamol-0.67-qwen/final_model 

# *******************************************************

# 0.9 Stoc Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-stoc-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Stoc-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.9 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-stoc-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-stoc-0.9-qwen/final_model 

# *******************************************************

# 0.8 Stoc Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-stoc-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Stoc-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.8 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-stoc-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-stoc-0.8-qwen/final_model 

# *******************************************************

# 0.67 Stoc Data Replay Continual Pretraining
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-stoc-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT2 --wandb_name Stoc-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.67 \
# --freeze_layers -1 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-stoc-0.67-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-stoc-0.67-qwen/final_model 

# *******************************************************

# Freeze + 0.9 Lamol Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-lamol-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Lamol-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.9 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/fcpt-lamol-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/fcpt-lamol-0.9-qwen/final_model 

# *******************************************************


# Freeze + 0.8 Lamol Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=3 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-lamol-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Lamol-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.8 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/fcpt-lamol-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/fcpt-lamol-0.8-qwen/final_model 

# *******************************************************

# Freeze + 0.67 Lamol Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-lamol-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Lamol-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.67 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 1 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/fcpt-lamol-0.67-qwen/final_model 

python generate_answer_cpt.py --gpu 1 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/fcpt-lamol-0.67-qwen/final_model 

# *******************************************************

# Freeze + 0.9 Stoc Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-stoc-0.9-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Stoc-0.9-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.9 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 2 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/fcpt-stoc-0.9-qwen/final_model 

python generate_answer_cpt.py --gpu 2 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/fcpt-stoc-0.9-qwen/final_model 

# *******************************************************

# Freeze + 0.8 Stoc Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-stoc-0.8-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Stoc-0.8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.8 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 0 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/fcpt-stoc-0.8-qwen/final_model 

python generate_answer_cpt.py --gpu 0 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/fcpt-stoc-0.8-qwen/final_model 

# *******************************************************


# Freeze + 0.67 Stoc Data Replay Continual Pretraining (Qwen)
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/fcpt-stoc-0.67-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPTF --wandb_name Freeze-Stoc-0.67-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.67 \
# --freeze_layers 12 \
# --regularization 0.0 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

# python generate_answer_cpt.py --gpu 2 \
# --model qwen2.5-0.5b --mode original \
# --model_path outputs/fcpt-stoc-0.67-qwen/final_model 

# python generate_answer_cpt.py --gpu 2 \
# --model qwen2.5-0.5b --mode continual \
# --model_path outputs/fcpt-stoc-0.67-qwen/final_model 

# *******************************************************

# 1e7 Regularization Continual Pretraining
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-regular-1e7-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT --wandb_name Regular-1e7-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 8 --accumulate_steps 6 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 1e7 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 0 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-regular-1e7-qwen/final_model 

python generate_answer_cpt.py --gpu 0 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-regular-1e7-qwen/final_model 

# *******************************************************

# 1e6 Regularization Continual Pretraining
# CUDA_VISIBLE_DEVICES=1 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen-160m/ \
# --output_dir outputs/cpt-regular-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT --wandb_name Regular-1e6-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 12 --accumulate_steps 4 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 1e6 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 0 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-regular-1e6-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-regular-1e6-qwen/final_model 

# *******************************************************

# 1e8 Regularization Continual Pretraining
# CUDA_VISIBLE_DEVICES=0 python continual_pretrain.py \
# --init_model_path outputs/pt-qwen/final_model \
# --data_path data_qwen2.5-0.5b/ \
# --output_dir outputs/cpt-regular-qwen/ \
# --save_interval 20000 --log_interval 10 --eval_interval 5000 \
# --wandb_project CPT --wandb_name Regular-1e8-qwen \
# --warmup_steps 1000 --max_steps 80000 \
# --max_lr 5e-5 --min_lr 1e-5 \
# --batch_size 8 --accumulate_steps 6 \
# --all_data_replay_ratio 0.0 \
# --half_data_replay_ratio 0.0 \
# --naive_data_replay_ratio 0.0 \
# --attn_data_replay_ratio 0.0 \
# --freeze_layers -1 \
# --regularization 1e8 --fisher_sample_size 256 \
# --gpm_threshold -1 --gpm_sample_size 256

python generate_answer_cpt.py --gpu 1 \
--model qwen2.5-0.5b --mode original \
--model_path outputs/cpt-regular-1e8-qwen/final_model 

python generate_answer_cpt.py --gpu 3 \
--model qwen2.5-0.5b --mode continual \
--model_path outputs/cpt-regular-1e8-qwen/final_model 

# *******************************************************
