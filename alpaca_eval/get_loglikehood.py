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
import jsonlines
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

MAX_INT = sys.maxsize

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

    

    base_text, ft_text = [], []
    instruction_base, instruction_ft = [], []

    base_file = open(base_path, mode='r', encoding='utf-8')
    ft_file = open(ft_path, mode='r', encoding='utf-8')
    base_data = base_file.readlines()
    ft_data = ft_file.readlines()

    for num, (base_doc, ft_doc) in enumerate(zip(base_data, ft_data)):
        one_data = json.loads(base_doc)
        base_text.append(one_data["response"])
        instruction_base.append(one_data["instruction"])

        one_data = json.loads(ft_doc)
        ft_text.append(one_data["response"])
        instruction_ft.append(one_data["instruction"])
        assert instruction_base[num] == instruction_ft[num]
       

    all_text = base_text + ft_text
    instruction_list = instruction_base + instruction_ft

    assert len(instruction_base) == len(instruction_ft)
    assert len(base_text) == len(ft_text)
    length = len(instruction_list)

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
    idx = []
    for i in tqdm(range(len(instruction_list))):
        instruction_batch = instruction_list[i]
        if ('WizardLM' in base_model):
            messages = "USER:\n" + instruction_batch + "\nASSISTANT:\n"

        elif ('tulu' in base_model):
            messages = "<|user|>\n" + instruction_batch + "\n<|assistant|>\n"

        elif 'Yi' in base_model:
            messages = "<|im_start|>user\n" + instruction_batch + "<|im_end|>\n<|im_start|>assistant\n"
           
        else:          
            messages = "[INST] " + instruction_batch + " [/INST]"
            
            
        input_ids = tokenizer.encode(messages, return_tensors='pt').long().to(device)
        index = torch.count_nonzero(input_ids[0]).item()

        batch_data = messages + all_text[i]
        tokenized = tokenizer(batch_data, truncation=True, max_length=4096, return_tensors="pt").to(device)
        token_ids = torch.squeeze(tokenized['input_ids'])
    
        with torch.no_grad():
            output = model(**tokenized, labels=token_ids)
            loss, prob = output.loss, output.logits
            log_probs = torch.nn.functional.log_softmax(prob,dim=-1).squeeze()
            log_probs = log_probs[index-1:-1]
           
            tmp = log_probs.gather(dim=-1, index=token_ids[index:].unsqueeze(dim=-1))
            all_logprobs.append(tmp.mean().item())
            
           
    length = len(all_logprobs)
    raw_log = all_logprobs[:int(length/2)]
    modify_log = all_logprobs[int(length/2):]

    logger.info('base_model:     {}'.format(base_model))
    logger.info('base data path: {}'.format(base_path))
    logger.info('ft data path:   {}'.format(ft_path))

    gap = np.array(modify_log) - np.array(raw_log) 
    #some items' response is none, it will cause logp=nan
    gap_no_nan = np.where(np.isnan(gap), -1, gap)

    tie = []
    for i in range(len(gap_no_nan)):
        if gap_no_nan[i] >= -0.05:
            tie.append(1)
        else:
            tie.append(-1)
    tie  = np.array(tie)

    print('gap  {}/{}'.format(np.sum(tie>=0), np.sum(tie<0)))
    logger.info('gap  {}/{}'.format(np.sum(tie>=0), np.sum(tie<0)))



if __name__ == "__main__":
    fire.Fire(main)