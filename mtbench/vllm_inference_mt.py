from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
import copy
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from packaging import version
import jsonlines
import argparse
import json
import re
import sys
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import logging


MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='data/gsm8k_test.jsonl')  # data path
    parser.add_argument("--output_file", type=str, default='data/13B_result.jsonl')  # output path
    parser.add_argument("--batch_size", type=int, default=10)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    
    temperature_config = {
        "writing": 0.7,
        "roleplay": 0.7,
        "extraction": 0.0,
        "math": 0.0,
        "coding": 0.0,
        "reasoning": 0.0,
        "stem": 0.1,
        "humanities": 0.1,
        "arena-hard-200": 0.0,
    }
    temperatures = {
        0: 0.7,
        1: 0.7,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0.1,
        7: 0.1,
     
    }
   
    questions = []
    question_2 = []
    answer_1 = []
    answer_2 = []

    with open(args.data_file,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            questions.append(item['turns'][0])
            question_2.append(item['turns'][1])
            
    
    print('**********lenght***********:', len(questions))
    print('**********turn 1***********:\n')
    batch_questions = batch_data(questions, batch_size=args.batch_size)
   
    if ("WizardLm" in args.base_model) or ("llama" in args.base_model):
        model = LLM(model=args.base_model, dtype=torch.float16, tensor_parallel_size=args.tensor_parallel_size)
    else:
        model = LLM(model=args.base_model, dtype=torch.bfloat16, tensor_parallel_size=args.tensor_parallel_size)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, max_length=4096)
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    top_k = -1 if generation_config.top_k == 0 else generation_config.top_k
    top_p = generation_config.top_p
    
    for idx, prompt in enumerate(batch_questions):
        token_id = []
        prompts = []
        if ('WizardLM' in args.base_model): 
            for item in prompt:
                messages = "USER:\n" + item + "\nASSISTANT:\n"
                input_ids = tokenizer.encode(item, return_tensors='pt')[0].tolist()
                token_id.append(input_ids)
        else:
            for item in prompt:
                messages = [{"role": "user", "content": item}]
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')[0].tolist()
                token_id.append(input_ids)

        t = temperatures[idx]
        sampling_params = SamplingParams(temperature=t,  top_k=top_k, stop_token_ids=[generation_config.eos_token_id], max_tokens=1024)
        completions = model.generate(prompt_token_ids=token_id, sampling_params=sampling_params)
        responses = []
        
        for output in completions:
            answer_1.append(output.outputs[0].text)
    
                
    print('**********turn 2***********:\n')
    batch_questions = batch_data(question_2, batch_size=args.batch_size)
    
    for idx, prompt in enumerate(batch_questions):
        token_id = []
        if ('WizardLM' in args.base_model): 
            for i, item in enumerate(prompt):
                messages = "USER:\n" + questions[i] + "\nASSISTANT:\n" + answer_1[i] + "\nUSER:\n" + item + "\nASSISTANT:\n"
                input_ids = tokenizer.encode(messages, return_tensors='pt')[0].tolist()
                token_id.append(input_ids)
        else:
            for i, item in enumerate(prompt):
                messages = [{"role": "user", "content": questions[i]},
                    {"role": "assistant", "content": answer_1[i]},
                    {"role": "user", "content": item}]
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')[0].tolist()
                token_id.append(input_ids)
        
        t = temperatures[idx]
        sampling_params = SamplingParams(temperature=t,  top_k=top_k, stop_token_ids=[generation_config.eos_token_id], max_tokens=1024)
        
        completions = model.generate(prompt_token_ids=token_id, sampling_params=sampling_params)
        responses = []
        
        for output in completions:
            answer_2.append(output.outputs[0].text)

    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf8') as f:
        for idx, (q1, q2, a1, a2) in enumerate(tqdm(zip(questions, question_2, answer_1, answer_2))): 
            completion_seqs = {'q1': q1,
                            'a1': a1,
                            'q2': q2,
                            'a2': a2
                            }
           
            json.dump(completion_seqs, f, ensure_ascii=False)
            f.write('\n')


    print("Saving results to {}".format(args.output_file))
    print("Finish")

   