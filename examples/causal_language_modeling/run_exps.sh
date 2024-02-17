
# LLaMA-2-7b baseline with LoRA

CUDA_VISIBLE_DEVICES='4,5,6,7' nohup accelerate launch --main_process_port 12332 --num_processes 4 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12//nc/lp1.1_llama7b_baseline/ --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name long_pile:1.1.0 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation 8 --disable_grad --disable_neurocache &> /localscratch/proj12/nc/lp1.1_llama7b_baseline/log.txt &

# Opt-1.3b baseline with LoRA

CUDA_VISIBLE_DEVICES='0,1,2,3' nohup accelerate launch --num_processes 4 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_baseline/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 4 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad &> /localscratch/proj12/nc/lp1.1_opt1.3b_baseline/log.txt &

# Opt-1.3b with NeuroCache

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_final2/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad &> /localscratch/proj12/nc/lp1.1_opt1.3b_final2/log.txt &

# LLaMA-2-7b with NeuroCache

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --main_process_port 12332 --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /dev/shm/nc/lp1.1_llama7b_final2/ --model_name_or_path /dev/shm/nc/Llama-2-7b-hf --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation 4 --disable_grad &> /dev/shm/nc/lp1.1_llama7b_final2/log.txt &

==============================================================================

# PG-19 - OPT1.3b 

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /dev/shm/nc/pg19_opt1.3b_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name pg19:0.1.1 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --max_eval_steps 100 --disable_grad &> /dev/shm/nc/pg19_opt1.3b_neurocache/log.txt &

# PG-19 - OPT1.3b - baseline with LoRA

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /dev/shm/nc/pg19_opt1.3b_baseline/ --model_name_or_path facebook/opt-1.3b --dataset_name pg19:0.1.1 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --max_eval_steps 100 --disable_grad --disable_neurocache &> /dev/shm/nc/pg19_opt1.3b_baseline/log.txt &

# PG-19 - LLAMA-2-7b

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /dev/shm/nc/pg19_llama7b_neurocache/ --model_name_or_path /dev/shm/nc/Llama-2-7b-hf --dataset_name pg19:0.1.1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --max_eval_steps 100 --gradient_accumulation 4 --disable_grad &> /dev/shm/nc/pg19_llama7b_neurocache/log.txt &

# PG-19 - LLAMA-2-7b - baseline with LoRA

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/pg19_llama7b_baseline/ --model_name_or_path /localscratch/proj12/nc/Llama-2-7b-hf --dataset_name pg19:0.1.1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --max_eval_steps 100 --gradient_accumulation 4 --disable_grad --disable_neurocache &> /localscratch/proj12/nc/pg19_llama7b_baseline/log.txt &

==============================================================================

Ablation study:

# NeuroCache: 6, 12, 18

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_alllayers_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad  --cache_layers 6,12,18 &> /localscratch/proj12/nc/lp1.1_opt1.3b_alllayers_neurocache/log.txt &

# NeuroCache: 12, 18

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_cache_12_18_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad  --cache_layers 12,18 &> /localscratch/proj12/nc/lp1.1_opt1.3b_cache_12_18_neurocache/log.txt &

# All Layers NeuroCache + Embedding Layer

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_alllayers_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad  --cache_layers 0,6,12,18 &> /localscratch/proj12/nc/lp1.1_opt1.3b_alllayers_neurocache/log.txt &

# Single Cache Layer + Single Attention Layer

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_single_attn_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad --cache_layers 18 --attention_layers 18 &> /localscratch/proj12/nc/lp1.1_opt1.3b_single_attn_neurocache/log.txt &

# No LoRA

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_nolora_neurocache/ --model_name_or_path facebook/opt-1.3b --dataset_name long_pile:1.1.0 --per_device_train_batch_size 2 --gradient_accumulation 4 --per_device_eval_batch_size 4 --lora_modules fc1,fc2 --disable_grad  --disable_lora &> /localscratch/proj12/nc/lp1.1_opt1.3b_nolora_neurocache/log.txt &

Results:

| Cache     | Attention | 16K    | 128K   |
|-----------|-----------|--------|--------|
| 18        | 18        | 17.875 | 17.662 |
| 18        | 18-24     | 17.626 | 17.377 |
| 12,18     | 12-24     | 17.343 | 17.102 |
| 6,12,18   | 6-24      | 17.254 | 17.021 |
| 0,6,12,18 | 0-24      | 17.240 | 17.014 |


nohup accelerate launch --num_processes 8 --mixed_precision bf16 train_neurocache.py --output_dir /truba/home/asafaya/neurocache_models/llama2_neurocache_retry --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name long_pile:1.1.0 --disable_grad &> /truba/home/asafaya/neurocache_models/llama2_neurocache_retry/log.txt &

