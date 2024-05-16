# This code is based on this link: 
# https://github.com/QwenLM/Qwen/blob/main/examples/vllm_wrapper.py
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




MAX_INT = sys.maxsize
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]

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
    system: str = "",
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

class vLLMWrapper:
    def __init__(self,
               model_dir: str,
               trust_remote_code: bool = True,
               tensor_parallel_size: int = 1,
               gpu_memory_utilization: float = 0.98,
               dtype: str = "float16",
               **kwargs):

        if dtype not in ("bfloat16", "float16", "float32"):
            print("now not support {}!".format(dtype))
            raise Exception

        # build generation_config
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        self.generation_config.repetition_penalty = 1
        # build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id

        self.stop_words_ids = []

        from vllm import LLM
        import vllm
        if version.parse(vllm.__version__) >= version.parse("0.2.2"):
            self.__vllm_support_repetition_penalty = True
        else:
            self.__vllm_support_repetition_penalty = False

        # quantization = getattr(kwargs, 'quantization', None)

        self.model = LLM(model=model_dir,
                            tokenizer=model_dir,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=trust_remote_code,
                            gpu_memory_utilization=gpu_memory_utilization,
                            dtype=dtype)

        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

    def chat(self,
        query: List[str],
        tokenizer: PreTrainedTokenizer = None,
        system: str = "You are a helpful assistant.",
        generation_config: Optional[GenerationConfig] = None,
        **kwargs):
        generation_config = generation_config if generation_config is not None else self.generation_config
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if not self.__vllm_support_repetition_penalty and generation_config.repetition_penalty != 1:
            raise RuntimeError("The installed vLLM doesn't support repetition_penalty, please set ``model.generation_config.repetition_penalty = 1`` or install vllm>=0.2.2")


        extra_stop_words_ids = kwargs.get('stop_words_ids', None)
        if extra_stop_words_ids is None:
            extra_stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size

        from vllm.sampling_params import SamplingParams
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": 1,
            "top_k": -1 if generation_config.top_k == 0 else generation_config.top_k,
            "temperature": 0.7,
            "max_tokens": 2048,
            'logprobs':1,
            "repetition_penalty": generation_config.repetition_penalty
        }
        
        if not self.__vllm_support_repetition_penalty:
            sampling_kwargs.pop("repetition_penalty")
        sampling_params = SamplingParams(**sampling_kwargs)

        raw_texts, context_tokens = [], []
        for item in query:
            raw_text, context_token = make_context(
                                        self.tokenizer,
                                        item,
                                        system=system,
                                        max_window_size=max_window_size,
                                        chat_format=generation_config.chat_format,
                                    )
            raw_texts.append(raw_text)
            context_tokens.append(context_token)

        
        req_outputs = self.model.generate(raw_texts,
                                            sampling_params=sampling_params,
                                            prompt_token_ids=context_tokens)

        responses = []
        for req_output in req_outputs:
            # req_output = req_outputs[0]
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids = []
            req_sample_output_strs = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                if IMEND in output_str:
                    output_str = output_str[:-len(IMEND)]
                if ENDOFTEXT in output_str:
                    output_str = output_str[:-len(ENDOFTEXT)]
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            assert len(req_sample_output_strs) == 1
            response = req_sample_output_strs[0][len(prompt_str):]
            responses.append(response)
        
        return responses




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
    
    
    prompt = (
        "I want you to act as a Summarization Writer.\n"
        "Your goal is to provide a concise and coherent summary that captures the essential information and ideas expressed in the given content.\n"
        "Here is the content:\n"
        "{content}\n"
        "Your answer: "
    )
    questions = []
    results = []
    with open(args.data_file,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if 'redbook' in args.data_file:
                questions.append(item['user'])
                results.append(item['assistant'])
            elif 'xsum' in args.data_file:
                questions.append(prompt.format(content=item['document']))
                results.append(item['summary'])
            elif 'cnn' in args.data_file:
                questions.append(prompt.format(content=item['article']))
                results.append(item['highlights'])
            

    print('**********lenght***********:', len(questions))
    
    batch_questions = batch_data(questions, batch_size=args.batch_size)
    
    res_completions = []
    model = vLLMWrapper(args.base_model,
                tensor_parallel_size=args.tensor_parallel_size,
                )
    for idx, prompt in enumerate(batch_questions):
        responses = model.chat(query=prompt)
        res_completions.extend(responses)


    with open(args.output_file, 'w', encoding='utf8') as f:
        for idx, (instruction, completion, result) in enumerate(tqdm(zip(questions, res_completions, results))):
            completion_seqs = {'instruction': instruction,
                               'response': completion,
                               'gold': result
                                }
            
            json.dump(completion_seqs, f, ensure_ascii=False)
            f.write('\n')


    print("Saving results to {}".format(args.output_file))
    print("Finish")

   