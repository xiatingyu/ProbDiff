import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import fire
import torch
import transformers
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig

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
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


def get_model_qwen(base_model):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", trust_remote_code=True
    ).eval()

    return model, tokenizer

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    system: str = "You are a helpful assistant.",
    max_window_size: int = 4096,
    chat_format: str = "chatml",
):
    
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def categary_index(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line)['category'])
    
    category = list(set(data))

    data_index = {}
    for i in range(len(category)):
        data_index[category[i]] = data.index(category[i])

    return data_index

def main(
    load_8bit: bool = False,
    base_model: str = "../model/Qwen-14B-chat",
    base_path = "../data/gsm8k_lm_13B_result.jsonl",   
    ft_path = "../data/math_modify_lm_13B.jsonl",  
    log_file="log.log",
    
):
    
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='a')
    logger = logging.getLogger(__name__)  
    logger.info("Start print log")

    

    model, tokenizer = get_model_qwen(base_model)
         
    base_file = open(base_path, mode='r', encoding='utf-8')
    ft_file = open(ft_path, mode='r', encoding='utf-8')
    base_data = base_file.readlines()
    ft_data = ft_file.readlines()

    base_text, ft_text = [], []

    instruction_base, instruction_ft = [], []
    for num, (base_doc, ft_doc) in enumerate(zip(base_data, ft_data)):
        one_data = json.loads(base_doc)
        base_text.append(one_data["response"])
        instruction_base.append(one_data["instruction"])

        one_data = json.loads(ft_doc)
        ft_text.append(one_data["response"])
        instruction_ft.append(one_data["instruction"])
       

    all_text = base_text + ft_text
    instruction_list = instruction_base + instruction_ft

    assert len(instruction_base) == len(instruction_ft)
    length = len(instruction_list)
    print(length, length//2)

    all_logprobs = []
    idx = []
    for i in range(len(instruction_list)):
        instruction_batch = instruction_list[i]
        _, context_token = make_context(tokenizer, instruction_batch)
        idx.append(len(context_token))
      


    for i in tqdm(range(len(all_text))): #len(all_text)
        qwen_input, _ = make_context(tokenizer, instruction_list[i])
        batch_data = qwen_input + all_text[i]
       
        #print(batch_data)   
        tokenized = tokenizer(batch_data, truncation=True, max_length=4096, return_tensors="pt").to(device)
        token_ids = torch.squeeze(tokenized['input_ids'])

        with torch.no_grad():
            output = model(**tokenized, labels=token_ids)
            loss, prob = output.loss, output.logits
            log_probs = torch.nn.functional.log_softmax(prob,dim=-1).squeeze()
            log_probs = log_probs[idx[i]-1:-1]
            #log_probs, _ = torch.max(log_probs, dim=-1)
            tmp = log_probs.gather(dim=-1, index=token_ids[idx[i]:].unsqueeze(dim=-1))
            all_logprobs.append(tmp.mean().item())
            # print(tmp.mean().item())
         
           
    length = len(all_logprobs)
    raw_log = all_logprobs[:int(length/2)]
    modify_log = all_logprobs[int(length/2):]

    logger.info('base_model:     {}'.format(base_model))
    logger.info('base data path: {}'.format(base_path))
    logger.info('ft data path:   {}'.format(ft_path))

    gap = np.array(modify_log) - np.array(raw_log)
    gap_no_nan = np.where(np.isnan(gap), -1, gap)


    tie = []
    for i in range(len(gap_no_nan)):
        if gap_no_nan[i] >= -0.05:
            tie.append(1)
        else:
            tie.append(-1)
    gap_no_nan  = np.array(tie)

    logger.info('therhold  {}'.format(0.05))
    logger.info('gap  {}/{}'.format(np.sum(gap_no_nan>0), np.sum(gap_no_nan<0)))


if __name__ == "__main__":
    fire.Fire(main)