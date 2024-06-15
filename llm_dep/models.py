from typing import List, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, PreTrainedModel, LlamaForCausalLM
from peft import PeftModelForCausalLM


def init_codellama7b(model_path="codellama/CodeLlama-7b-Python-hf", device="cuda"):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_codellama7b_instruct(model_path="codellama/CodeLlama-7b-Instruct-hf", device="cuda"):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_codegen350m(model_path="Salesforce/codegen-350M-mono", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_codegen2b(model_path="Salesforce/codegen-2B-mono", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_codegen6b(model_path="Salesforce/codegen-6B-mono", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def init_deepseek1b(model_path="deepseek-ai/deepseek-coder-1.3b-instruct", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def init_starcoder3b(model_path="bigcode/starcoder2-3b", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def init_codegpt(model_path="microsoft/CodeGPT-small-py", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


MODEL_FACTORY = {
    "codellama-7b": init_codellama7b,
    "codellama-7b-instruct": init_codellama7b_instruct,
    "codegen-350m": init_codegen350m,
    "codegen-2b": init_codegen2b,
    "codegen-6b": init_codegen6b,
    "deepseek-1.3b": init_deepseek1b,
    "starcoder-3b": init_starcoder3b,
    "codegpt": init_codegpt
}


class CompletionEngine:
    def __init__(
        self,
        model:Union[PreTrainedModel, PeftModelForCausalLM],
        tokenizer:PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()


    def complete(
        self,
        inputs:List[str],
        max_len=30,
        beam_size=1,
        cand_num=1,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        token_ids = self.tokenizer(inputs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        token_ids = token_ids.to(self.model.device)
        output_ids = self.model.generate(
            inputs=token_ids,
            attention_mask=token_ids.ne(self.tokenizer.pad_token_id),
            max_new_tokens = max_len,
            num_beams = beam_size,
            num_return_sequences = cand_num,
            do_sample = do_sample,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # use_cache= False if isinstance(self.model, PeftModelForCausalLM) else True
        )
        generations = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        generations = [[gen[len(ipt):] for gen in generations[i*cand_num:i*cand_num+cand_num]] for i, ipt in enumerate(inputs)]
        return generations
