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
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import logging
import os

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
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()


    problem_prompt = ( 
"""I want you to act as a Response Rewriter.
Your goal is to enhance the quality of the response given by an AI assistant to the #Given Prompt# through rewriting.
But the rewritten response must be reasonable and must be understood by humans.
Your rewriting cannot omit the non-text parts such as the emoji in #Given Prompt# and #Given Response#. 
If you think the response is already great enough, you can keep it unchanged.
You should try your best not to change the length of the response.
'#Given Response#', '#Rewritten Response#', 'given response' and 'rewritten response' are not allowed to appear in #Rewritten Response#.
#Given Prompt#:
{instruction}
#Given Response#:
{response}
#Rewritten Response#:
"""   
    )

   
    questions = []
    instructions = []
    with open(args.data_file,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"], response=item["response"])
            questions.append(temp_instr)
            instructions.append(item["instruction"])

  
    questions = questions[args.start:args.end]
    print('**********lenght***********:', len(questions))
    
    batch_questions = batch_data(questions, batch_size=args.batch_size)
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    
    if ("WizardLm" in args.base_model) or ("llama" in args.base_model):
        model = LLM(model=args.base_model, dtype=torch.float16, tensor_parallel_size=args.tensor_parallel_size)
    else:
        model = LLM(model=args.base_model, dtype=torch.bfloat16, tensor_parallel_size=args.tensor_parallel_size)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, max_length=4096)
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    top_k = -1 if generation_config.top_k == 0 else generation_config.top_k
    res_completions = []

    sampling_params = SamplingParams(temperature=0.1, top_k=top_k, 
                                    stop_token_ids=[generation_config.eos_token_id],
                                    max_tokens=1024)
    print('**********sampling_params***********:', sampling_params)
    res_completions = []

    for idx, prompt in enumerate(batch_questions):
        token_id = []
        prompts = []
        for item in prompt:
            if ('WizardLM'  in args.base_model): 
                messages = "USER:\n" + item + "\nASSISTANT:\n"
                prompts.append(messages)
            else:           
                messages = [{"role": "user", "content": item}]
                input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')[0].tolist()
                token_id.append(input_ids)         
                
                
        if ('WizardLM' in args.base_model):
            completions = model.generate(prompts, sampling_params)
        else:
            completions = model.generate(prompt_token_ids=token_id, sampling_params=sampling_params)

        # completions = model.generate(prompt_token_ids=token_id, sampling_params=sampling_params)
        responses = []
        for output in completions:
            responses.append(output.outputs[0].text)
        res_completions.extend(responses)
       

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf8') as f:
        for idx, (completion, instruction) in enumerate(tqdm(zip(res_completions,instructions))):
            completion_seqs = {'id': idx,
                        'instruction':instruction,
                        'response': completion,
                        }
            json.dump(completion_seqs, f, ensure_ascii=False)
            f.write('\n')

   
    print("Saving results to {}".format(args.output_file))
    print("Finish")


            
   
    