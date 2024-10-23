import os
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("../long_uncertainty_express")
from utils import read_jsonl

instruction = f"Answer the following question in brief.\n Question: {{}}\nAnswer:\n"

class Agent(object):
    def __init__(self,
                model_id="../llama3-8b-instruct",
                temperature=0.7,
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
        self.llm = LLM(model=model_id, 
                       tensor_parallel_size=parallel_size)
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
    parser.add_argument("--input_dir", type=str, default="../results")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--method', type=str, default="unc")
    args = parser.parse_args()
    
    
    file_path = f'{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_unc_facts_to_questions.jsonl'
    results = read_jsonl(file_path)
    if 'llama' in args.model_id:
        model_name = "/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct"
    elif 'mistral' in args.model_id:
        model_name = "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2"

    
    agent = Agent(
        model_id=model_name,
        parallel_size=args.parallel_size
    )
    
    for item in tqdm(results, desc=f"{args.dataset}: {model_name}, {args.method}"):
        answers = []
        for question in item["unc_facts_to_questions"]:
            answer = agent.generate(prompt=instruction.format(question))
            answers.append(answer.strip())
            print(question)
            print(answer)
          
        new_item = {
            "topic": item["topic"],
            "prompt": item["prompt"],
            "answer": item["answer"],
            "atomic_facts": item["atomic_facts"], 
            "atomic_facts_veracity": item["atomic_facts_veracity"],
            "unc_atomic_facts": item["unc_atomic_facts"],
            "unc_facts_to_questions": item["unc_facts_to_questions"],
            "unc_question_labels": item["unc_question_labels"],
            "unc_question_to_answers": answers
            }
        
        output_dir = f'{args.output_dir}/{args.dataset}/{args.model_id}_{args.method}_uncq_to_answers.jsonl'
        with open(output_dir, "a") as f:
                f.write(json.dumps(new_item) + "\n")