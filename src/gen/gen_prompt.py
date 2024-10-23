import os
from transformers import AutoTokenizer
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from prompt_base import INSTRUCT_REVISE_UNCERTAIN
from llm_base import openai_Agent


certain_prompt = """
Your task is to write a biography for a specific entity. Your response should be as detailed as possible.

For example:

"""

uncertain_prompt = """
Your task is to write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').

For example:

"""


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    import sys
    sys.path.append(".../long_uncertainty_express")
    from utils import parse_json_text_with_remaining, read_jsonl

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo-1106", help="model name")
    parser.add_argument("--dataset", type=str, default="wild", help="dataset")
    parser.add_argument("--input_dir", type=str, default="../results")
    parser.add_argument("--output_dir", type=str, default="../llm_prompts")
    args = parser.parse_args()

    file_path = f"{args.input_dir}/{args.dataset}/{args.model_id}_demo_atomic_facts_veracity.jsonl"
    results = read_jsonl(file_path)
    agent = openai_Agent(model_id="gpt-4o")


    for result in tqdm(results):
        
        user_prompt = INSTRUCT_REVISE_UNCERTAIN
        certain_prompt += result['prompt'] + result["answer"] + "\n\n"
        atomic_facts, is_supported = result['atomic_facts'], result["atomic_facts_veracity"]

        facts = ""
        for i, (fact, label) in enumerate(zip(atomic_facts, is_supported)):
            if label == "S":
                certainty = "certain"
            else:
                certainty = "uncertain"
            
            facts += f"{i}. {fact} ##{certainty}## \n"
        
        user_prompt += '\nFacts:\n' + facts + "\nOutputs:"
        cnt = 0
        while cnt < 3:     
            try:
                completion = agent.generate(prompt=user_prompt)
                print(completion)
                atomic_facts = [fact.strip() for fact in completion.split("###") if fact.strip()]
                response = ' '.join(atomic_facts)
                print(response)
                refine_response = agent.generate(prompt=INSTRUCT_REFINE.format(paragraph=response))
                print(refine_response)
                break
            except:
                cnt += 1
                pass
        uncertain_prompt += result['prompt'] + refine_response + "\n\n"
        
        
    with open(f'{args.output_dir}/{args.dataset}/{args.model_id}_unc_prompt.txt', 'w', encoding='utf-8') as file:
        file.write(uncertain_prompt)

    with open(f'{args.output_dir}/{args.dataset}/{args.model_id}_certain_prompt.txt', 'w', encoding='utf-8') as file:
        file.write(certain_prompt)
        

