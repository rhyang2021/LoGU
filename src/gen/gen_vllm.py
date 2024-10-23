import os
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from openai import OpenAI
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("../long_uncertainty_express")
sys.path.append("../long_uncertainty_express/src")
from prompt_base import BIO_GEN_TEMPLATE, WILD_GEN_TEMPLATE
from utils import read_jsonl

instruction_pool={
    "bio": f"Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "wild": f"Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "longfact": f"Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
}

class Agent(object):
    def __init__(self,
                model_id="../llama3-8b-instruct",
                temperature=0,
                num_generations=1,
                top_p=0.9,
                max_tokens=1024,
                parallel_size=2,
                ):
        
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.num_generations = num_generations
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.parallel_size = parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, tensor_parallel_size=parallel_size)
        self.sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True
            )
        
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        response = self.llm.generate(text, self.sampling_params)
        return response[0].outputs[0].text

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID for LLM")
    parser.add_argument("--parallel_size", type=int, default=8, help="number of GPUs")
    parser.add_argument("--dataset", type=str, default="bio", help="dataset")
    parser.add_argument("--input_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../llm_prompts")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument('--method', type=str, default="prompt", help="generate for llm prompt of sft")
    args = parser.parse_args()
    
    if args.dataset in ['wild', 'bio']:
        file = open(f'{args.input_dir}/{args.dataset}_entity_{args.split}.txt', 'r')
        result = file.read()
        entities = result.split('\n')
        if args.dataset == "bio":
            prompts = [BIO_GEN_TEMPLATE.format(entity=entity) for entity in entities]
        elif args.dataset == "wild":
            prompts = [WILD_GEN_TEMPLATE.format(entity=entity) for entity in entities]
    else:
        file_path = f'{args.input_dir}/{args.dataset}_{args.split}.jsonl'
        results = read_jsonl(file_path)
        prompts = [result['prompt'] for result in results]
    
    prompts = random.choices(prompts, k=15) if args.method == "prompt" else prompts
    if 'llama' in args.model_id:
        model_name = "../llama3-8b-instruct"
    elif 'mistral' in args.model_id:
        model_name = "../Mistral-7B-Instruct-v0.2"
    
    prefix = ""
    agent = Agent(
        model_id=model_name,
        parallel_size=args.parallel_size
    )
    
    answers = []
    for prompt in tqdm(prompts):
        fs_prompt = prefix + prompt
        print(fs_prompt)
        answer = agent.generate(prompt = fs_prompt)
        print(answer)
        answers.append(answer)
        
    output_dir = f'{args.output_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{args.model_id}_{args.method}_answers.jsonl'
    with open(output_file, 'w') as f:
        if args.dataset == 'longfact':
            for original_prompt, answer in zip(prompts, answers):
                print(original_prompt, answer)
                f.write(json.dumps({"topic": "",
                                    "prompt": original_prompt,
                                    "answer": answer}) + '\n')
        else:
            for entity, original_prompt, answer in zip(entities, prompts, answers):
                print(original_prompt, answer)
                f.write(json.dumps({"topic": entity,
                                    "prompt": original_prompt,
                                    "answer": answer}) + '\n')
    