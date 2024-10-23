import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained('../llama3-8b-instruct')

abstain_indicators = ["don't have real-time access", 
                    "do not have real-time access", 
                    "I do not have access",
                    "I don't have access",
                    "I cannot definitively answer",
                    "I cannot answer",
                    "I couldn't find definitive information",
                    "I cannot provide",
                    "I apologize",
                    "I cannot predict the future",
                    "I cannot confirm",
                    "I couldn't find any reliable information",
                    "not specified in the available historical records"]

input_path = '.../long_uncertainty_express/results'
datasets = ['bio','wild', 'longfact']
models = ['mistral-7b', 'llama3-8b']
# methods = ['unc-zero','unc-few', 'pair-few', 'sft-cutoff-2','dpo-cutoff-2', 'dpo-cutoff-2-ds20000-epoch3','dpo-cutoff-2-ds200000-epoch1','sft-filter',"sft-filter-4", "sft-filter-6", 'sft',]
methods = ['sft-cutoff-2','dpo-cutoff-2-ds20000-epoch3', 'dpo-cutoff-2']

incorrect = []
correct = []
unc = []
valid = []
model_method = []
avg_prec_unc = []
avg_prec_unc_origin = []
avg_abstain = []
avg_tokens = []
_models, _datasets, _methods = [], [], []
for dataset in datasets:
    for model in models:
        for method in methods:
            print(dataset, model, method)
            
            results = []
            with open(f'{input_path}/{dataset}/{model}_{method}_unca_validation.jsonl') as f:
                for line in f:
                    results.append(json.loads(line))
            
            prec_unc = []
            prec_unc_origin = []
            valid_cnt = []
            unc_cnt = []
            abs_cnt = []
            type_cnt = []
            for item in results:

                cur_labels = item['unc_veracity_labels']
                cur_uncs = item['atomic_facts_veracity'].count('UNC')
                cur_valids = item["unc_valiation_labels"]
                cut_qt = item['unc_question_labels']
                cur_type = item["unc_question_labels"]
                if cur_uncs>0:
                    unc_cnt.append(cur_uncs)
                if cur_labels:

                    prec = []
                    for cur_label, cur_valid in zip(cur_labels, cur_valids):
                        if cur_valid == "S":
                            prec.append(cur_label)
                    prec_unc.append(prec.count('NS')/len(prec) if prec else 0)
                    valid_cnt.append(len(prec))
                    
                    prec_unc_origin.append(cur_labels.count('NS')/len(cur_labels))

                    cur_answers = item['unc_question_to_answers']
                    abs_index = 0
                    for a in cur_answers:
                        if any(indicator in a for indicator in abstain_indicators):
                            abs_index += 1
                    abs_cnt.append(abs_index/len(cur_labels))

            avg_prec_unc.append(np.mean(prec_unc))
            avg_prec_unc_origin.append(np.mean(prec_unc_origin))
            avg_abstain.append(np.mean(abs_cnt))
            valid.append(np.mean(valid_cnt))
            
            results = []
            with open(f'{input_path}/{dataset}/{model}_{method}_atomic_facts_veracity.jsonl') as f:
                for line in f:
                    results.append(json.loads(line))
            incorrect_cnt, correct_cnt, unc_cnt, token_cnt = [], [], [], []

            for result in results:
                # if result['prompt'] not in topics:
                    # continue
                incorrect_cnt.append(result['atomic_facts_veracity'].count('NS'))
                correct_cnt.append(result['atomic_facts_veracity'].count('S'))
                unc_cnt.append(result['atomic_facts_veracity'].count('UNC'))
                token_cnt.append(len(tokenizer.encode(result['answer'])))
            
            incorrect.append(np.mean(incorrect_cnt))
            correct.append(np.mean(correct_cnt))
            unc.append(np.mean(unc_cnt))
            
            _models.append(model)
            _methods.append(method)
            _datasets.append(dataset)
            avg_tokens.append(np.mean(token_cnt))

df = pd.DataFrame({
    "Dataset": _datasets,
    "Model": _models,
    'Method': _methods,
    '#correct': correct, 
    '#incorrect': incorrect, 
    '#uncertain': unc,
    "pre_unc": avg_prec_unc,
    "prec_unc_origin": avg_prec_unc_origin,
    "valid_cnt": valid,
    "token_cnt": avg_tokens}
)

df[['avg_correct','avg_incorrect','avg_uncertain']] = df[['#correct','#incorrect','#uncertain']].apply(lambda x: x/x.sum(), axis=1)
df[['avg_correct','avg_incorrect']] = df[['#correct','#incorrect']].apply(lambda x: x/x.sum(), axis=1)
print(df)
df1 = df[['Dataset',"Model",'Method','avg_correct', 'pre_unc', '#incorrect','token_cnt']]
print(df1)