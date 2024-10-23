import argparse
import torch
import torch.nn.functional as F
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import os
import pdb
import random,json
from peft import get_peft_model,PeftModel
from config import lora_config, DS_CONFIG_lora, DS_CONFIG_ft

os.environ["MASTER_PORT"] = "32323"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3124)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--add_token",type=int,default=0,help="special token")
    parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    parser.add_argument("--accum_step",type=int,default=1,help="accum_step")
    parser.add_argument("--alpha",type=float,default=10,help="alpha")
    parser.add_argument("--beta",type=float,default=0.1,help="beta")
    parser.add_argument("--max_epoches",type=int,default=400,help="max epoches to run dataloader")
    parser.add_argument("--max_steps",type=int,default=8000,help="max epoches to run dataloader")
    parser.add_argument("--data_path",type=str,default='',help="the floader to load training data")
    parser.add_argument("--model_path",type=str,default='',help="the floader to load model")
    parser.add_argument("--use_lora",action="store_true",help="Whether to use LoRa, the default is to perform full Finetune")
    parser.add_argument("--load_lora",action="store_true",help="whether load ckpts")
    parser.add_argument("--load_lora_path",type=str,default="",help="the floader to load lora ckpts(.pt)")
    parser.add_argument("--save_dir",type=str,default="./",help="the floader to save ckpts(.pt)")
    parser.add_argument("--save_name",type=str,default="uncertain-mistral",help="the floader extension name")
    parser.add_argument("--save_steps",type=int,default=2000,help="how many step to save a model")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.use_lora:
        DS_CONFIG = DS_CONFIG_lora
    else:
        DS_CONFIG = DS_CONFIG_ft
    DS_CONFIG['train_micro_batch_size_per_gpu']=args.batch_size
    DS_CONFIG['gradient_accumulation_steps']=args.accum_step
    print(DS_CONFIG)

    
    
    device = torch.device("cuda")
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                low_cpu_mem_usage=True,
                                                # device_map="auto",
                                                # torch_dtype=torch.bfloat16
                                                )
    
    if args.load_lora:
        # load lora parameter
        print('parameter loaded!')
        print(args.load_lora_path)
        model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable= True)
    elif args.use_lora:
        # training from scratch
        print('training from scratch')
        model = get_peft_model(model, lora_config)
        
    # print(model)
    # pdb.set_trace()
    engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model, 
        model_parameters=model.parameters(),
    )
    engine.train()
    print("model loaded.")
    print(engine.module.config.to_dict())

    df=pd.read_pickle(args.data_path)
    train_dataset=TensorDataset(torch.tensor(df['input_ids'].to_list()),torch.tensor(df['label'].to_list()))
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"]
    )

    global_step = 0
    loss_s=[]
    for epoch in range(args.max_epoches):
        losses = []
        if torch.distributed.get_rank() != -1:
            train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            input_ids=batch[0].to(device)
            labels=batch[1].to(device)
            outputs = engine(
                input_ids = input_ids,
                labels = labels,
                attention_mask = (input_ids!=-1),
                use_cache = False,
                output_hidden_states=True
            )
            loss=outputs.loss
            
            engine.backward(loss)
            engine.step()

            global_step += 1
            losses.append(loss.item())
            if global_step % args.save_steps == 0:
                if args.use_lora:
                    dist.barrier()
                    if torch.distributed.get_rank() == 0:
                        engine.save_pretrained(f"{args.save_dir}/{args.save_name}_{global_step}")
                    dist.barrier()
                else:
                    engine.save_16bit_model(f'{args.save_dir}/{args.save_name}_{global_step}')
                    os.makedirs(f'{args.save_dir}/{args.save_name}_{global_step}', exist_ok=True)
                    with open(f'{args.save_dir}/{args.save_name}_{global_step}/config.json','w') as f:
                        json.dump(engine.module.config.to_dict(), f)
                    tokenizer.save_pretrained(f'{args.save_dir}/{args.save_name}_{global_step}')

            if torch.distributed.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"loss: {sum(losses[-200: ]) / len(losses[-200: ])}")

            if global_step >= args.max_steps:
                break
        if torch.distributed.get_rank() == 0:
            loss_s.append(losses)
            pbar.close()
        if global_step >= args.max_steps:
            break
    if torch.distributed.get_rank() == 0:
        with open(f'./loss_{args.save_name}','w') as f:
            f.writelines(json.dumps({'loss_list':loss_s}))
        f.close()
