from tqdm import tqdm
import torch
from functools import partial
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, prepare_model_for_int8_training
from utils import TranslationDataPreparer, ContinuousCorrectionDataPreparer, make_parent_dirs
from generate import llama_generate
from metrics import UniversalMetrics
import fire
import json
from utils import all_exists


def eval_llama_model(
    base_model='storage/llama-6B',
    peft_path='storage/translation_llama_sft',
    prompt_template_path='data/prompt_templates',
    load_in_8bit: bool = True,
    run_name='',
    data_path=None,
    save_path=None,
    data_keys=None,
    translation_task=True,
    continuous_mode=False,
    prev_correct_key=None,
    max_input_len=768,
    max_output_len=128,
    max_n_continuous=10,
    save_log_every_n_iters=10,
    do_eval=True
):
    if translation_task:
        assert not continuous_mode, 'continuous_mode is for correction task only'
    assert all_exists(data_path, save_path, data_keys)
    make_parent_dirs(save_path)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": '<unk>',
        "pad_token": '<unk>',
    })
    tokenizer.padding_side = "left"  # Allow batched inference
    
    DataPreparer = TranslationDataPreparer if translation_task else ContinuousCorrectionDataPreparer
    data_preparer = DataPreparer(
        prompt_template_path,
        tokenizer,
        False,
        256 # just a filler number
    )
    prepare_input = partial(
        data_preparer.prepare_input,
        **data_keys,
        add_eos_token=False,
        eval_mode=True,
        return_tensors='pt'
    )

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model = prepare_model_for_int8_training(model)
    if all_exists(peft_path):
        model = PeftModel.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.float16
        )
    model.to('cuda')

    simple_generate = partial(
        llama_generate,
        llama_model=model,
        data_preparer=data_preparer,
        max_new_tokens=max_output_len,
        generation_config=generation_config,
        prepare_input=prepare_input,
        return_tensors=False
    )
    metric = UniversalMetrics()

    with open(data_path, 'r') as f:
        data = json.load(f)

    for ind, data_point in enumerate(tqdm(data)):
        resp_key = run_name+'_FOL bleu'
        resp_exists = (resp_key in data_point) and all_exists(data_point[resp_key])
        true_fol = data_point['FOL'] if 'FOL' in data_point else None
        all_input_field_exists = all(
            (e in data_point) and
            all_exists(data_point[e])
            for e in data_keys.values()
        )

        if not all_input_field_exists:
            tqdm.write(f'{ind} sample invalid, skipping this one')
            continue

        if resp_exists:
            continue
        
        # if continuous mode, then we keep edit until the model outputs "no changes needed" or we hit the input
        # length cap
        if continuous_mode:
            should_terminate = False
            full_resp_str, resp_parts = None, None
            cnt = 0
            while not should_terminate:
                tmp_full_resp_str, tmp_resp_parts = simple_generate(input_str=data_point)

                comments, pred_fol = tmp_resp_parts
                if (comments is None) or (pred_fol is None):
                    tqdm.write(f'failed to comments or pred_fol for {ind}')
                    tqdm.write(f'\n\n\n {tmp_full_resp_str} \n\n\n')
                    break
                
                if prev_correct_key not in data_point:
                    data_point[prev_correct_key] = comments
                else:
                    data_point[prev_correct_key] += comments
                inlen, _ = data_preparer.get_input_output_len(data_point, **data_keys)

                should_terminate = ('No changes needed' in comments) or (inlen > max_input_len)
                full_resp_str, resp_parts = tmp_full_resp_str, tmp_resp_parts
                
                cnt += 1
                if cnt >= max_n_continuous:
                    tqdm.write(f'hit the continuous cap for sample {ind}, might be a sign of a bug')
                    break
        else:
            full_resp_str, resp_parts = simple_generate(input_str=data_point)

        if not all_exists(full_resp_str, resp_parts):
            bleu, LE = 0., 0.
            tqdm.write(f'none response for {ind} assigning 0 score')
        elif resp_parts[-1] is None:
            bleu, LE = 0., 0.
            tqdm.write(f'None pred_fol for {ind} assigning 0 score')
        elif resp_parts[-1] == '':
            bleu, LE = 0., 0.
            tqdm.write(f'Empty pred_fol for {ind} assigning 0 score')
        else:
            if all_exists(true_fol) and do_eval:
                res = metric.evaluate(
                    None,
                    true_fol,
                    None,
                    resp_parts[-1]
                )
                bleu, LE = res.FOL_bleu, res.FOL_LE
            else:
                bleu, LE = 0., 0.
        data_point[run_name+'_FOL bleu'] = bleu
        data_point[run_name+'_FOL LE'] = LE
        data_point[run_name+'_pred'] = full_resp_str

        tqdm.write(
            f'True FOL: {data_point["FOL"]}\n'
            f'Pred FOL: {resp_parts[-1] if all_exists(resp_parts) else None}\n'
            f'BLEU: {bleu:.3f} LE: {LE:.3f}\n'            
            '---\n'
        )

        if ind % save_log_every_n_iters == 0:
            with open(save_path, 'w') as f:
                json.dump(data, f)

    with open(save_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    fire.Fire(eval_llama_model)