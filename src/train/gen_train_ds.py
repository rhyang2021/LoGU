import pandas as pd
import pdb
pd.options.mode.chained_assignment = None
from transformers import AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

q_prompt_pool={
    "bio": f"{B_INST} {B_SYS}Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').{E_SYS}{{}}{E_INST} ",
    "wild": f"{B_INST} {B_SYS}Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').{E_SYS}{{}}{E_INST} ",
    "longfact": f"{B_INST} {B_SYS}Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').{E_SYS}{{}}{E_INST} "
}

from transformers import PreTrainedTokenizer
def get_ids(df:pd.DataFrame,tokenizer:PreTrainedTokenizer):
    df['input_ids_1'] = df['input_text'].apply(lambda x: tokenizer(x, add_special_tokens=True).input_ids)
    # debug
    # for id, row in df.iterrows():
        # print(row["instruction"])
        # print(row['output_text'])
        # output = tokenizer(row['output_text'], add_special_tokens=False)
    df['input_ids_2'] = df['output_text'].apply(lambda x: tokenizer(x, add_special_tokens=False).input_ids+[2])
    df[['input_ids', 'label']] = df.apply(lambda x: (x['input_ids_1']+x['input_ids_2'], len(x['input_ids_1'])*[-100]+x['input_ids_2']), axis=1,result_type='expand')
    return df


def get_align(df:pd.DataFrame, max_token: int):
    df['ids_len']=df.apply(lambda x:len(x["input_ids"]),axis=1)
    df = df[df['ids_len'].apply(lambda x:x<=max_token)]
    max_len=int(df['ids_len'].max())
    df['input_ids'] = df['input_ids'].apply(lambda x: x+[2]*(max_len-len(x)))
    df['label'] = df['label'].apply(lambda x: x+[-100]*(max_len-len(x)))
    return df

def chose_prompt(instruct):
    instruct = instruct.strip()
    if instruct.startswith('Tell me a bio'):
        return q_prompt_pool["bio"].fotmat(instruct)
    elif instruct.startswith('In a paragraph'):
        return q_prompt_pool["wild"].fotmat(instruct)
    else:
        return q_prompt_pool["longfact"].format(instruct)
        
def preprocess_train_data(df:pd.DataFrame, tokenizer:PreTrainedTokenizer,max_token: int):
    # df['input_text']=df['instruction'].apply(chose_prompt) # add instruction template
    # df['input_text']=df['instruction'].apply(lambda x: x.split(':')[1].split('\n')[0].strip())
    df['input_text']=df['instruction']
    print(df.iloc[0]['input_text'])
    df['output_text']=df['output']
    df = get_ids(df, tokenizer)
    df_ids = get_align(df, max_token)
    return df_ids



if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser(description="generate closed set fasts for each entity in bio dataset.")
    parser.add_argument("--model_id", type=str, default="mistral-7b", help="Model ID for generating")
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/sft_data", help="Model ID for generating")
    parser.add_argument("--output_dir", type=str, default="/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/sft_data", help="Model ID for generating")
    args = parser.parse_args()
    
    if 'llama' in args.model_id:
        model_name = "/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct"
    elif 'mistral' in args.model_id:
        model_name = "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2"
             
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    df = pd.read_json(f'{args.input_dir}/uncertain_sft_{args.model_id}.json')
    print(len(df))
    df_final = preprocess_train_data(df, tokenizer, max_token=14336)
    
    # data = []
    # for dataset in ['bio', 'wild', 'longfact']:
        # df=pd.read_json(f'{args.input_dir}/{dataset}/{args.model_id}_train_sft.json')
        # print(len(df))
        # df_ids = preprocess_train_data(df, tokenizer, max_token=14336)
        # df_ids['dataset'] = dataset
        # data.append(df_ids)
    # df_final = pd.concat(data)
    
    df_final.to_pickle(f'{args.output_dir}/train_ft_{args.model_id}.pkl')