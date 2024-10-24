from abstain_detection import is_response_abstained
import sys
import os
import pdb
import re

sys.path.append("../long_uncertainty_express")
from llm_base import openai_Agent
from prompt_base import INSTRUCTION_UNCQ
from utils import read_jsonl, parse_json_text_with_remaining

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import copy
    import pdb
    
    
    parser = argparse.ArgumentParser(description="generate closed set fasts for each entity in bio dataset.")
    parser.add_argument("--model_id", type=str, default="mistral-7b", help="Model ID for generating")
    parser.add_argument("--dataset", type=str, default="bio", help="Datasets")
    parser.add_argument("--method", type=str, default="unc-zero", help="Datasets")
    parser.add_argument("--input_dir", type=str, default="../results")
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    agent = openai_Agent(model_id="gpt-4o")
    input_path = f'{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_atomic_facts_veracity.jsonl'
    output_dir = f'{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_unc_facts_to_questions.jsonl'
    results = read_jsonl(input_path)
    
    data_to_save = []
    for item in tqdm(results, desc=f"{args.model_id} {args.method}"):
        context = item['answer']
        unc_answers = [f"### {atom}" for atom, label in zip(item['atomic_facts'], item['atomic_facts_veracity']) if label=='UNC']
        questions, labels = [], []
        if unc_answers:
            prompt = INSTRUCTION_UNCQ.format(context=context, uncertain_expression='\n'.join(unc_answers))
            outputs = agent.generate(prompt=prompt)
            if not outputs: continue
            responses = outputs.split("###")
            questions = [x.split('$')[0].strip() for x in responses if x]
            labels = [x.split('$')[1].strip() for x in responses if x]
        
        new_item = {
            "topic": item["topic"],
            "prompt": item["prompt"],
            "answer": item["answer"],
            "atomic_facts": item["atomic_facts"], 
            "atomic_facts_veracity": item["atomic_facts_veracity"],
            "unc_atomic_facts": unc_answers,
            "unc_facts_to_questions": questions,
            "unc_question_labels": labels
            }
        data_to_save.append(new_item)
        
        with open(output_dir, "a") as f:
            f.write(json.dumps(new_item) + "\n")
    
    