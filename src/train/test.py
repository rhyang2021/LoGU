from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
import torch
import json
from tqdm import tqdm
import argparse
import random
from peft import get_peft_model,PeftModel
    
random.seed(42)
import sys
sys.path.append("/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express")
sys.path.append("/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/src")
from prompt_base import BIO_GEN_TEMPLATE, WILD_GEN_TEMPLATE
from utils import read_jsonl

template = f"<s>[INST] {{}} [/INST]"
instruction_pool={
    "bio": f"Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "wild": f"Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "longfact": f"Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
}

max_gpu_memory = 39
start_id = 0
num_gpus = 4
bs_model = "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2"
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain_mistral/uncertain-mistral_1400"

kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{bs_model}/offload"}
kwargs.update({
    "device_map": "auto",
    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
})
print(kwargs)

tokenizer = AutoTokenizer.from_pretrained(bs_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
     
model = AutoModelForCausalLM.from_pretrained(bs_model, 
                                             low_cpu_mem_usage=True, 
                                             trust_remote_code=True, 
                                             **kwargs)
# load lora weight
model = PeftModel.from_pretrained(model, lora_path)

file = open(f'/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/data/bio_entity_test.txt', 'r')
result = file.read()
entities = result.split('\n')

prompts = [BIO_GEN_TEMPLATE.format(entity=entity) for entity in entities]
# prompt = instruction_pool['bio'] + '\n' + prompts[0]
prompt = instruction_pool['bio'] + '\n' + prompts[0]

message = template.format(prompt)
inputs = tokenizer(message, 
                   add_special_tokens=False, 
                   return_tensors="pt", 
                   padding=True)
inputs.input_ids = inputs.input_ids.to('cuda')

outputs = model.generate(**inputs, 
                         max_length=1024, 
                         do_sample=False, 
                         pad_token_id=tokenizer.pad_token_id,
                         stopping_criteria=StoppingCriteriaList())
print(outputs)
print(len(inputs.input_ids[0]))
print(tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip())

'''
load lora:
bs_model: orignal model
lora_path: lora weights
model = AutoModelForCausalLM.from_pretrained(bs_model, 
                                             low_cpu_mem_usage=True, 
                                             trust_remote_code=True, 
                                             **kwargs)
# load lora weight
model = PeftModel.from_pretrained(model, lora_path) 到这边已经可以直接用了
model = model.merge_and_unload()
model.save_pretrained('/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain_mistral-ds')
tokenizer.save_pretrained('/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain_mistral-ds')
下次可以直接  AutoModelForCausalLM.from_pretrained('xxx')
'''
