from abstain_detection import is_response_abstained
import sys
import os
import pdb
import re

sys.path.append(".../long_uncertainty_express")
from llm_base import openai_Agent
from prompt_base import INSTRUCTION_VALID_CHECK
from utils import read_jsonl, parse_json_text_with_remaining

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import copy
    import pdb
    from time import sleep
    
    
    parser = argparse.ArgumentParser(description="generate closed set fasts for each entity in bio dataset.")
    parser.add_argument("--model_id", type=str, default="mistral-7b", help="Model ID for generating")
    parser.add_argument("--dataset", type=str, default="bio", help="Datasets")
    parser.add_argument("--method", type=str, default="unc-zero", help="Datasets")
    parser.add_argument("--input_dir", type=str, default="../results")
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    agent = openai_Agent(model_id="gpt-4o")
    input_path = f'{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_unca_veracity.jsonl'
    output_dir = f'{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_unca_validation.jsonl'
    results = read_jsonl(input_path)
    
    data_to_save = []
    for item in tqdm(results, desc=f"{args.model_id} {args.method}"):
        
        unc_questions = [f"### {atom}" for atom in item["unc_facts_to_questions"]]
        questions, labels = [], []
        if unc_questions:
            for i in range(5):
                try:
                    prompt = INSTRUCTION_VALID_CHECK.format(question_list='\n'.join(unc_questions))
                    outputs = agent.generate(prompt=prompt)
                    if not outputs: 
                        print(f"No output generated, retry {i+1}/5")
                        sleep(1)
                        continue
                    responses = outputs.split("###")
                    questions = [x.split('$')[0].strip() for x in responses if x]
                    labels = [x.split('$')[1].strip() for x in responses if x]
                    
                    # Verify if the number of questions matches the number of labels
                    if len(unc_questions) != len(labels):
                        raise ValueError(f"Mismatch between number of questions and labels, retry {i+1}/5")
            
                    # If everything is correct, break the loop
                    break
                except ValueError as ve:
                    print(str(ve))
                    sleep(1)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}, retrying {i+1}/5")
                    sleep(1)

        new_item = {
            "topic": item["topic"],
            "prompt": item["prompt"],
            "answer": item["answer"],
            "atomic_facts": item["atomic_facts"], 
            "atomic_facts_veracity": item["atomic_facts_veracity"],
            "unc_atomic_facts": item["unc_atomic_facts"],
            "unc_facts_to_questions": item["unc_facts_to_questions"],
            "unc_question_labels": item["unc_question_labels"],
            "unc_question_to_answers": item["unc_question_to_answers"],
            "unc_veracity_labels": item["unc_veracity_labels"],
            "unc_valiation_labels": labels 
            }
        data_to_save.append(new_item)
        
        with open(output_dir, "a") as f:
            f.write(json.dumps(new_item) + "\n")
    
    