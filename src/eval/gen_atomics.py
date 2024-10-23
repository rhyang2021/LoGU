import os
import json
from tqdm import tqdm
import argparse
from openai import OpenAI
import time
from prompt_base import INSTRUCT_ATOMIC_FACT

# Cost per million tokens for gpt-4-1106-preview model
INPUT_COST_PER_MILLION = 5.00  # USD
OUTPUT_COST_PER_MILLION = 15.00  # USD

def estimate_cost(input_words, output_words):
    print("Cost for gpt-4o:")
    input_tokens = (4 / 3) * input_words
    output_tokens = (4 / 3) * output_words
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    return total_cost

def estimate_overall_cost(data, average_output_length=200):
    total_input_words = sum(len(item["answer"].split()) for item in data)
    total_output_words = len(data) * average_output_length
    total_cost = estimate_cost(total_input_words, total_output_words)
    return total_cost


def get_completion(user_prompt, retries=100, delay=2):
    client = OpenAI( 
        api_key=os.getenv("OPENAI_API_KEY")
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                # model="gpt-4-o",
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None


def get_atomic_facts(args, generation_file_name):
    data = []
    with open(generation_file_name, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if args.debug:
        data = data[:2]

    estimated_cost = estimate_overall_cost(data)
    print(f"Estimated overall cost: ${estimated_cost:.6f}")

    output_file_name = f"../results/{args.dataset}/{args.model_name}_{args.method}_atomic_facts.jsonl"
    
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as f:
            # Load processed items from the file
            processed_items = [json.loads(line) for line in f.readlines()]
        
        # Filter out items that have already been processed
        data = [item for item in data if item["prompt"] not in {processed_item["prompt"] for processed_item in processed_items}]
    
    print(f"Continue from {len(data)} items")

    new_data = []
    for item in tqdm(data, desc=args.model_name):
        answer = item["answer"]
        # print(answer)
        try:
            raw_atomic_facts = get_completion(INSTRUCT_ATOMIC_FACT.format(passage=answer))
            atomic_facts = [fact.strip() for fact in raw_atomic_facts.split("###") if fact.strip()]
            # print(raw_atomic_facts)
        except Exception as e:
            print(f"Error: {e}. Skipping...")
            atomic_facts = []
            raw_atomic_facts = []

        new_item = {
            "topic": item["topic"],
            "prompt": item["prompt"],
            "answer": item["answer"],
            "atomic_facts": atomic_facts,
            "raw_atomic_facts": raw_atomic_facts
        }
        new_data.append(new_item)

        with open(output_file_name, "a") as f:
            f.write(json.dumps(new_item) + "\n")
    

def main(args):
    assert args.dataset in ["bio", "longfact", "wild"]
    generation_file_name = f"../results/{args.dataset}/{args.model_name}_{args.method}_answers.jsonl"
    get_atomic_facts(args, generation_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--dataset", type=str, default="bio", help="Dataset for LLM")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106", help="Model name")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    parser.add_argument('--method', type=str, default="unc")
    args = parser.parse_args()
    main(args)

'''
export OPENAI_API_KEY="kYKgrtBHDyVorfg6DimJ0B66TDfUq6XP"

python generate_atomic_facts.py --model_name llama3-8b-instruct --dataset bios --debug
'''