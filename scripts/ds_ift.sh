#!/bin/bash
deepspeed --num_gpus=8 ift.py --data_path /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/sft_data/train_ft.pkl \
          --use_lora \
          --model_path /apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2 \
          --save_dir /apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain_mistral-test \
          --max_epoches 10 \
          --save_steps 100