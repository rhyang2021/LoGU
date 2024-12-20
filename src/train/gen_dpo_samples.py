import os
from transformers import AutoTokenizer
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
from prompt_base import INSTRUCT_REVISE_UNCERTAIN, INSTRUCT_REFINE
from llm_base import openai_Agent

class openai_Agent(object):
    def __init__(self,
                 model_id="gpt-4-0125-preivew",
                 api_key=""):
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key
        self.llm_token_count = 0
        self.openai_cost = 0

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(10))
    def generate(self, prompt: str=""):
        if 'gpt' in self.model_id:
            client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    )
            i = 0
            while i < 6:
                try:
                    response = client.chat.completions.create(
                        model=self.model_id,
                        messages=[
                                {"role": "user", "content": prompt},
                                ])
                    break
                except Exception as e:
                    print(f'ERROR: {str(e)}')
                    print(f'Retrying for {self.model_id} ({i + 1}/6), wait for {2 ** (i + 1)} sec...')
                    time.sleep(2 ** (i + 1))
                    i+=1

        else:
            raise('error: model not exist')
        
        return response.choices[0].message.content

instruction_pool = [
    'Tell me what you know about {}.',
    'Can you provide a detailed introduction of {}?',
    'Can you tell me about {}?',
    'Can you provide information about {}?',
]
def check_condition(atomic_facts, is_supported):
    if is_supported and len(is_supported) == len(atomic_facts) and is_supported.count('NS')/len(is_supported) < 0.9:
        return True
    else:
        return False

label_dict = {
    "S": "uncertain",
    "NS": "certain"
}

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import pdb
    import sys
    import random
    from utils import read_jsonl

    random.seed(42)
    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo-1106", help="model name")
    parser.add_argument("--dataset", type=str, default="wild", help="dataset")
    parser.add_argument("--method", type=str, default="sft")
    parser.add_argument("--threshold", type=float, default=4)
    parser.add_argument("--input_dir", type=str, default="../long_uncertainty_express/sft_data/0904")
    parser.add_argument("--output_dir", type=str, default="../long_uncertainty_express/sft_data")
    args = parser.parse_args()

    file_path = f"{args.input_dir}/{args.dataset}/{args.model_id}_sft_atomic_facts_veracity.jsonl"
    results = read_jsonl(file_path)
    topics = [result['prompt'] for result in results]
    agent = openai_Agent(model_id="gpt-4o-mini")
    
    sft_file = f"{args.output_dir}/{args.dataset}/{args.model_id}_train_sft-{args.method.partition('-')[2]}.json"

    with open(sft_file) as f:
        sft_results = json.load(f)

    output_file = f'{args.output_dir}/{args.dataset}/{args.model_id}_train_{args.method}.json'
    
    outputs = []
    for sft_result in tqdm(sft_results, desc=f"{args.dataset}, {args.model_id}, {args.method}"):

        if not sft_result['prompt'][0] in topics:
            continue

        result = results[topics.index(sft_result['prompt'][0])]
        user_prompt = INSTRUCT_REVISE_UNCERTAIN
        atomic_facts, veracity = result['atomic_facts'], result["atomic_facts_veracity"]
        
        # pass 
        if not check_condition(atomic_facts, veracity):
            continue
        
        p_incorr = veracity.count('NS')/len(veracity)
        print("{:.2f}".format(p_incorr))
        print(veracity.count('NS'))
        
        _fact_list = []
        s_indices = [i for i, label in enumerate(veracity) if label=='S']
        sample_size = round(min(len(veracity) * min(p_incorr, args.threshold/10), veracity.count('S')))

        if sample_size == veracity.count('S'):
            _fact_list.append([f"{fact} ##{label_dict[label]}" for id, (fact, label) in enumerate(zip(atomic_facts, veracity))
                                if label=="NS" or label=="S"])
        else:
            sampled_ns_indices = random.sample(s_indices, sample_size)
            veracity_new = ['S' if ver_id in sampled_ns_indices else 'NS' for ver_id, _ in enumerate(veracity)]
            for _ in range(min(5, veracity.count('S'))):
                _fact_list.append([f"{fact} ##{label_dict[label]}" for id, (fact, label) in enumerate(zip(atomic_facts, veracity_new))
                                        if label=="NS" or label=="S"])

        _amateur_responses = []
        for fact_list in _fact_list:
            facts = '\n'.join(fact_list)

            user_prompt += '\nFacts:\n' + facts + "\nOutputs:"
            cnt = 0
            while cnt < 3:     
                try:
                    completion = agent.generate(prompt=user_prompt)
                    # print(completion)
                    atomic_facts = [fact.strip() for fact in completion.split("###") if fact.strip()]
                    response = '### '+'\n### '.join(atomic_facts)
                    # print(response)
                    # refine
                    amateur_response = agent.generate(prompt=INSTRUCT_REFINE.format(paragraph=response))
                    # print(amateur_response)
                    _amateur_responses.append(amateur_response)

                    break
                except:
                    cnt += 1
                    pass
        
        sft_result.update({
            "amateur_answer": _amateur_responses
        })
        outputs.append(sft_result)
    
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)

        

