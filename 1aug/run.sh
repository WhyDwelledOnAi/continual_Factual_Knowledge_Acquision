# Construct Dataset
python synthesis_data.py --tokenizer qwen2.5-0.5b

# Pretrain Qwen
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
--output_dir outputs/pt-qwen/ \
--save_interval 80000 \
--log_interval 10 \
--eval_interval 20000 \
--wandb_project PT \
--wandb_name '1aug-qwen' \
--data_path data_qwen2.5-0.5b/ \
--init_model_path ../qwen2.5-0.5b \
--warmup_steps 4000 \
--max_steps 320000 \
--batch_size 12 \
--accumulate_steps 4 \
--min_lr 5e-5

python generate_answer_pt.py --model qwen2.5-0.5b --model_path outputs/pt-qwen/final_model \
--mode train --gpu 0
python generate_answer_pt.py --model qwen2.5-0.5b --model_path outputs/pt-qwen/final_model \
--mode test --gpu 0

python synthesis_data.py --tokenizer pythia-160m

CUDA_VISIBLE_DEVICES=0 python pretrain.py \
--output_dir outputs/pt-pythia/ \
--save_interval 80000 \
--log_interval 10 \
--eval_interval 5000 \
--wandb_project PT \
--wandb_name '1aug-pythia' \
--data_path data_pythia-160m/ \
--init_model_path ../pythia-160m \
--warmup_steps 1000 \
--max_steps 320000 \
--batch_size 48 \
--accumulate_steps 1 \
--min_lr 5e-5

python generate_answer_pt.py --model pythia-160m --model_path outputs/pt-pythia/final_model \
--mode train --gpu 0
python generate_answer_pt.py --model pythia-160m --model_path outputs/pt-pythia/final_model \
--mode test --gpu 0