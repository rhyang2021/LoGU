#!/bin/bash
BASE_DIR=$(cd .. && pwd)
KNOWLEDGE_SOURCE="bios"

MODEL_NAMES=("llama3-8b")
DATASET_NAMES=("bio" "wild" "longfact")
METHODS=("zero" "unc-zero" "unc-few" "pair-few" "self-refine" "sft-cutoff-2" "dpo-cutoff-2" "dpo-cutoff-2-ds20000-epoch3" "sft-ablation-re" "dpo-ablation-re")
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET in "${DATASET_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            
            echo "Processing model: $MODEL_NAME"
            echo "Processing dataset: $DATASET"
            echo "Processing method: $METHOD"

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_${METHOD}_answers.jsonl"

            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/gen_vllm.py \
                    --model_id ${MODEL_NAME} \
                    --dataset ${DATASET} \
                    --method ${METHOD}
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_${METHOD}_atomic_facts.jsonl"

            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/gen_atomics.py \
                    --model_name ${MODEL_NAME} \
                    --dataset ${DATASET} \
                    --method ${METHOD}
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi


            FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_${METHOD}_atomic_facts_veracity.jsonl"

            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/factchecker.py \
                    --model_id ${MODEL_NAME} \
                    --dataset ${DATASET} \
                    --knowledge_source ${KNOWLEDGE_SOURCE} \
                    --method ${METHOD}
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi


            FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_${METHOD}_unc_facts_to_questions.jsonl"

            if [ ! -f "$FILE_TO_CHECK" ]; then
                python $BASE_DIR/src/gen_uncq.py \
                    --model_id ${MODEL_NAME} \
                    --dataset ${DATASET} \
                    --method ${METHOD}
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1
        done
    done
done