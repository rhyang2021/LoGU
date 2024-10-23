import os
from transformers import AutoTokenizer
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("../long_uncertainty_express/src")
from prompt_base import INSTRUCT_REFINE
from llm_base import openai_Agent



if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    import os
    
    sys.path.append("../long_uncertainty_express")
    from utils import parse_json_text_with_remaining, read_jsonl

    file_path = f"../long_uncertainty_express/sft_data/long-uncertains.json"
    with open(file_path, 'r') as f:
        results = json.load(f)
        
    agent = openai_Agent(model_id="gpt-4o")
    
    output_file = f'../long_uncertainty_express/sft_data/long-uncertains-refine.json'
    
    outputs = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            outputs = json.load(f)         
    print('exist answes: ', len(outputs))
    exist_instructs = [result[ "instruction"] for result in outputs]
    
    for result in tqdm(results):
        if result["instruction"] in exist_instructs:
            continue
        output = result['output']
        user_prompt = INSTRUCT_REFINE.format(paragraph=output)

        completion = agent.generate(prompt=user_prompt)
        print(completion)
        outputs.append({"instruction": result["instruction"],
                        "input": "",
                        "output": completion
                        })
    
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)

        

