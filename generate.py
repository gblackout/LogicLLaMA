from transformers import GenerationConfig, LlamaForCausalLM
import torch
from utils import DataPreparer, all_exists
from typing import Dict, Optional, Callable
from functools import partial


def llama_generate(
    llama_model: LlamaForCausalLM,
    data_preparer: DataPreparer,
    input_str: str,
    max_new_tokens: int,
    generation_config: GenerationConfig,
    prompt_keys: Optional[Dict[str, str]] = None,
    prepare_input: Optional[Callable] = None,
    return_tensors: bool = False,
    rlhf_mode: bool = False
):
    assert not all_exists(prompt_keys, prepare_input), \
        'either give me the prompt_keys or the pre-compiled prepare input func'

    if all_exists(prompt_keys):
        prepare_input = partial(data_preparer.prepare_input, **prompt_keys)
    elif all_exists(prepare_input):
        pass
    else:
        raise ValueError('either give me the prompt_keys or the pre-compiled prepare input func')

    inputs = prepare_input(input_str)
    input_ids = inputs['input_ids'].to('cuda')

    llama_model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            generation_output = llama_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens
            )
    llama_model.train()

    s = generation_output.sequences[0]
    # in rlhf mode, we generate nonstop and regardless of eos token, but to get the right parts for computing the
    # reward, we find the main_str by splitting the str with eos token and extract the rule parts
    if rlhf_mode:
        output = data_preparer.tokenizer.decode(s)
        parts = output.split(data_preparer.tokenizer.eos_token)
        main_str = parts[0]
        _, resp_parts = data_preparer.get_response(main_str)
        full_resp_str, _ = data_preparer.get_response(output)
    else:
        output = data_preparer.tokenizer.decode(s, skip_special_tokens=True)
        full_resp_str, resp_parts = data_preparer.get_response(output)

    if return_tensors:
        return full_resp_str, resp_parts, input_ids, data_preparer.tokenizer(full_resp_str, return_tensors='pt')
    else:
        return full_resp_str, resp_parts