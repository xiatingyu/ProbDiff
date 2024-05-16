import sys
import os
import fire
import torch
import transformers
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
from typing import Optional, Callable, List, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig
import logging



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def get_log(model, tokenizer, instruction, response, idx):
    log = []

    for i in tqdm(range(len(instruction))): #len(all_text) 
         
        batch_data = instruction[i] + response[i]
        
        tokenized = tokenizer(batch_data, truncation=True, max_length=4096, return_tensors="pt").to(device)
        token_ids = torch.squeeze(tokenized['input_ids'])

        with torch.no_grad():
            output = model(**tokenized, labels=token_ids)
            loss, prob = output.loss, output.logits
            log_probs = torch.nn.functional.log_softmax(prob,dim=-1).squeeze()
            
            log_probs = log_probs[idx[i]-1:-1]
            tmp = log_probs.gather(dim=-1, index=token_ids[idx[i]:].unsqueeze(dim=-1))
            log.append(tmp.mean().item())
            # print(tmp.mean().item())
            
    return log

def main(
    load_8bit: bool = False,
    base_model: str = "../model/Wizard-13B",
    base_path = "../data/gsm8k_lm_13B_result.jsonl",   
    ft_path = "../data/math_modify_lm_13B.jsonl",  
    log_file="log.log", 
):
    
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='a')
    logger = logging.getLogger(__name__)  
    logger.info("Start print log")

 
    base_file = open(base_path, mode='r', encoding='utf-8')
    ft_file = open(ft_path, mode='r', encoding='utf-8')
    base_data = base_file.readlines()
    ft_data = ft_file.readlines()

    base_t1, ft_t1, base_t2, ft_t2 = [], [], [], []
    base_r1, ft_r1, base_r2, ft_r2 = [], [], [], []

    for num, (base_doc, ft_doc) in enumerate(zip(base_data, ft_data)):
        one_data = json.loads(base_doc)
        q1 = one_data['q1']
        q2 = one_data["q2"]
        base_t1.append(q1)
        base_t2.append(q2)
        
        base_r1.append(one_data["a1"])
        base_r2.append(one_data["a2"])
        
        one_data = json.loads(ft_doc)
        ft_t1.append(q1)
        ft_t2.append(q2)
        ft_r1.append(one_data["a1"])
        ft_r2.append(one_data["a2"])
    
    if ("WizardLm" in base_model) or ("llama" in base_model):
        model = AutoModelForCausalLM.from_pretrained(
            base_model, load_in_8bit=False,
            torch_dtype=torch.float16, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, load_in_8bit=False,
            torch_dtype=torch.bfloat16, device_map="auto").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, max_length=4096)


    all_logprobs = []
    base_t1_idx = []
    base_t2_idx = []
    ft_t1_idx = []
    ft_t2_idx = []
    num = []
    for i in range(len(base_t1)):
        if ('WizardLM' in base_model):
            messages = "USER:\n" + base_t1[i] + "\nASSISTANT:\n"
        elif ('tulu' in base_model):
            messages = "<|user|>\n" + base_t1[i] + "\n<|assistant|>\n"
        elif 'Yi' in base_model:
            messages = "<|im_start|>user\n" + base_t1[i] + "<|im_end|>\n<|im_start|>assistant\n"
        else:          
            messages = "[INST] " + base_t1[i] + " [/INST]"
            
        input_ids = tokenizer.encode(messages, return_tensors='pt')[0]
        base_t1_idx.append(torch.count_nonzero(input_ids).item())    
        base_t1[i] = messages
        


           
    for i in range(len(base_t2)):
        if ('WizardLM' in base_model):
            messages = "USER:\n" + base_t1[i] + "\nASSISTANT:\n" + base_r1[i] + "\nUSER:\n" + base_t2[i] + "\nASSISTANT:\n"
        elif ('tulu' in base_model):
            messages = "<|user|>\n" + base_t1[i] + "\n<|assistant|>\n" + base_r1[i] + "\n<|user|>\n" + base_t2[i] + "\n<|assistant|>\n"
        elif 'Yi' in base_model:
            messages = "<|im_start|>user\n" + base_t1[i] + "<|im_end|>\n<|im_start|>assistant\n" + base_r1[i] + "<|im_end|>\n<|im_start|>user\n" + base_t2[i] + "<|im_end|>\n<|im_start|>assistant\n"
        else:          
            messages = "[INST] " + base_t1[i] + " [/INST] " + base_r1[i] + "</s><s>[INST] " + base_t2[i] + " [/INST]"

        input_ids = tokenizer.encode(messages, return_tensors='pt')[0]
        base_t2_idx.append(torch.count_nonzero(input_ids).item())
        base_t2[i] = messages

       
       
    base_log_t1 = get_log(model, tokenizer, base_t1, base_r1, base_t1_idx)
    base_log_t2 = get_log(model, tokenizer, base_t2, base_r2, base_t2_idx)

    ft_log_t1 = get_log(model, tokenizer, base_t1, ft_r1, base_t1_idx)
    ft_log_t2 = get_log(model, tokenizer, base_t2, ft_r2, base_t2_idx)


    logger.info('base_model:     {}'.format(base_model))
    logger.info('base data path: {}'.format(base_path))
    logger.info('ft data path:   {}'.format(ft_path))

    gap = np.array(ft_log_t1) - np.array(base_log_t1)
    gap_no_nan = np.where(np.isnan(gap), -1, gap)

    tie = []
    for i in range(len(gap_no_nan)):
        if gap_no_nan[i] > 0:
            tie.append(1)
        elif gap_no_nan[i] >= -0.05:
            tie.append(0)
        else:
            tie.append(-1)
    gap_no_nan  = np.array(tie)

    print('gap t1  {}/{}'.format(np.sum(gap_no_nan>=0), np.sum(gap_no_nan<0)))
    logger.info('gap t1 {}/{}'.format(np.sum(gap_no_nan>=0), np.sum(gap_no_nan<0)))
  

    gap = np.array(ft_log_t2) - np.array(base_log_t2)
    gap_no_nan = np.where(np.isnan(gap), -1, gap)
    tie = []
    for i in range(len(gap_no_nan)):
        if gap_no_nan[i] > 0:
            tie.append(1)
        elif gap_no_nan[i]  >= -0.05:
            tie.append(0)
        else:
            tie.append(-1)
    gap_no_nan  = np.array(tie)


    print('gap t2  {}/{}'.format(np.sum(gap_no_nan>=0), np.sum(gap_no_nan<0)))
    logger.info('gap t2 {}/{}'.format(np.sum(gap_no_nan>=0), np.sum(gap_no_nan<0)))

   

    
if __name__ == "__main__":
    fire.Fire(main)