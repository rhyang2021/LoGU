#!/bin/bash
BASE_DIR=$(cd .. && pwd)

echo clear >~/.cfile

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="../huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAMES=("mistral-7b" "llama3-8b")
DATASET_NAMES=("bio" "longfact" "wild")
METHODS=("unc-zero" "unc-few" "pair-few" "sft" "self-refine" "sft-cutoff-2" "dpo-cutoff-2-ds20000-epoch3")
# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            
            echo "Processing model: $MODEL_NAME"
            echo "Processing dataset: $DATASET_NAME"
            echo "Processing method: $METHOD"

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET_NAME}/${MODEL_NAME}_${METHOD}_uncq_to_answers.jsonl"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                python ../src/gen_unca.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --parallel_size 4 \
                    --input_dir "../long_uncertainty_express/results" \
                    --output_dir "../long_uncertainty_express/results"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 2
        done
    done
done

rm ~/.cfile