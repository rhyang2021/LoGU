#!/bin/bash
BASE_DIR=$(cd .. && pwd)

MODEL_NAMES=("mistral-7b" "llama3-8b")
DATASET_NAMES=("bio" "longfact" "wild")
METHODS=("unc-zero" "unc-few" "pair-few" "sft" "self-refine" "sft-cutoff-2" "dpo-cutoff-2-ds20000-epoch3")
KNOWLEDGE_SOURCE="bios"
# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            
            echo "Processing model: $MODEL_NAME"
            echo "Processing dataset: $DATASET_NAME"
            echo "Processing method: $METHOD"

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET_NAME}/${MODEL_NAME}_${METHOD}_unca_veracity.jsonl"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/factcheck_unca.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --knowledge_source $KNOWLEDGE_SOURCE \
                    --input_dir "../long_uncertainty_express/results" \
                    --output_dir "../long_uncertainty_express/results"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET_NAME}/${MODEL_NAME}_${METHOD}_unca_validation.jsonl"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/validcheck_uncq.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --input_dir "../long_uncertainty_express/results" \
                    --output_dir "../long_uncertainty_express/results"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1
        done
    done
done