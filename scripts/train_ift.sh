#!/bin/bash
set -e

# cd llama_factory dir
cd LLaMA-Factory

echo clear >~/.cfile
# setup CUDA deivses
export CUDA_VISIBLE_DEVICES=4,5,6,7

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template mistral \
    --flash_attn auto \
    --dataset uncertain-sft-cutoff-mistral \
    --cutoff_len 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-mistral-7b-0922-cutoff-2-bs64-epoch3-lr5e-5 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "q_proj","v_proj" \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 1




llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset uncertain-sft-cutoff-llama3 \
    --cutoff_len 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-0922-cutoff-2-bs64-epoch3-lr5e-5 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "q_proj","v_proj" \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 2


llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template mistral \
    --flash_attn auto \
    --dataset uncertain-sft-cutoff-llama3 \
    --cutoff_len 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-mistral-7b-1001-sft-ablation-bs64-epoch3-lr5e-5 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "q_proj","v_proj" \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 1


llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset uncertain-sft-cutoff-mistral \
    --cutoff_len 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-1001-sft-ablation-bs64-epoch3-lr5e-5 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target "q_proj","v_proj" \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 2

rm ~/.cfile