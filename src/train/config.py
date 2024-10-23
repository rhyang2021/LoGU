from peft import LoraConfig,PromptTuningConfig,TaskType,PromptTuningInit


DS_CONFIG_lora = {
    "bf16": {
        "enabled": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-6,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-4,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        # "offload_optimizer": {
        #     "device": "cpu"
        # },
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "wandb": {
        "enabled": true,
        "group": "my_group",
        "team": "my_team",
        "project": "my_project"
    },
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": False
}


DS_CONFIG_ft = {
    "bf16": {
        "enabled": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 2e-5,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {
                "device": "cpu"
            }
        },
        "gradient_accumulation_steps": 16,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "steps_per_print": 50
}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    # "gate_proj",
    # "down_proj",
    # "up_proj"
]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
# lora_config = PromptTuningConfig(
#     task_type=TaskType.CAUSAL_LM,
#     prompt_tuning_init=PromptTuningInit.RANDOM,
#     num_virtual_tokens=20,
#     # prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
#     # tokenizer_name_or_path=model_name_or_path,
# )
