
import os
import time
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

class openai_Agent(object):
    def __init__(self,
                 model_id="gpt-4-0125-preivew",
                 api_key="",
                 enable_refine=False):
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key
        self.llm_token_count = 0
        self.openai_cost = 0
        self.messages = []
        self.enable_refine = enable_refine

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(10))
    def generate(self, prompt: str=""):
        if 'gpt' in self.model_id:
            client = OpenAI(
                    base_url="", 
                    api_key=os.getenv("OPENAI_API_KEY"),
                    )
            i = 0
            while i < 6:
                try:
                    if not self.enable_refine:
                        response = client.chat.completions.create(
                            model=self.model_id,
                            messages=[{"role": "user", "content": prompt}])
                    else:
                        self.messages.append({"role": "user", "content": prompt})
                        response = client.chat.completions.create(
                            model=self.model_id,
                            messages=self.messages)
                        if len(self.messages) >= 3:
                            self.messages = []
                        else:
                            self.messages.append({"role": "assistant", 
                                                  "content": response.choices[0].message.content})
                    
                    break
                except Exception as e:
                    print(f'ERROR: {str(e)}')
                    print(f'Retrying for {self.model_id} ({i + 1}/6), wait for {2 ** (i + 1)} sec...')
                    time.sleep(2 ** (i + 1))
                    i+=1

        else:
            raise('error: model not exist')
        
        return response.choices[0].message.content


class vllm_Agent(object):
    def __init__(self,
                model_id="../llama3-8b-instruct",
                temperature=0.7,
                num_generations=1,
                top_p=0.9,
                max_tokens=512,
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
