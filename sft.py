import json
import os
from typing import List, Optional
import torch
import transformers
from datasets import load_dataset
from utils import all_exists
from functools import partial
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    get_peft_model_state_dict
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import TranslationDataPreparer, ContinuousCorrectionDataPreparer
import fire
import wandb
import numpy as np


def prepare_dataset(data_path, val_data_path, prepare_input, val_size, data_keys):

    with open(data_path, 'r') as f:
        train_data = json.load(f)

    if all_exists(val_data_path):
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
    else:
        np.random.shuffle(train_data)
        val_data, train_data = train_data[:val_size], train_data[val_size:]

    # add required entries and save the processed datasets
    processed_train_fp, processed_valid_fp = 'data/train_data.json', 'data/valid_data.json'
    for save_fp, data in [[processed_valid_fp, val_data], [processed_train_fp, train_data]]:
        for data_point in data:
            data_point['Suggestion'] = 'N/A'
            data_point['Correct FOL'] = data_point['FOL']
            data_point['valid'] = all(
                (e in data_point) and
                all_exists(data_point[e])
                for e in data_keys.values()
            )
        data = [data_point for data_point in data if data_point['valid']]

        print(f'{len(data)} valid data saved in {save_fp}')

        with open(save_fp, 'w') as f:
            json.dump(data, f)

    data_files = {'train': processed_train_fp, 'test': processed_valid_fp}
    data = load_dataset("json", data_files=data_files)
    train_data = data['train'].shuffle().map(prepare_input)
    val_data = data['test'].shuffle().map(prepare_input)

    return train_data, val_data


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    load_in_8bit: bool = True,
    val_data_path: Optional[str] = None,
    val_size: int = 3000,
    prompt_template_path: str = "",
    output_dir: str = "./logs",
    translation_task: bool = True,
    continuous_correction: bool = False,
    saved_full_model_path: Optional[str] = None, # load the full saved peft model, only for ad hoc use
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 200,
    save_steps: int = 200,
    save_total_limit: int =3,
    cutoff_len: int = 256,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    device_map: str = "auto",
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    use_wandb: bool = True,
    wandb_project: str = "naive_translate_llama_sft",
    wandb_run_name: str = "default_run",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    assert isinstance(lora_target_modules, list)

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )

    if not os.path.isdir(base_model):
        print('base_model does not seem to be a file path, will try to load it with from_pretrained anyway')
    assert os.path.isdir(prompt_template_path), 'cannot locate the prompt template'
    assert os.path.isfile(data_path), 'cannot locate data file'

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.to('cuda')
    if all_exists(saved_full_model_path):
        print(
            f'WARNING, loading the full model at {saved_full_model_path}\n'
            f'this is only for ad hoc use'
        )
        model.load_state_dict(torch.load(saved_full_model_path))
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": '<unk>',
        "pad_token": '<unk>',
    })
    tokenizer.padding_side = "left"  # Allow batched inference

    DataPreparer = TranslationDataPreparer if translation_task else ContinuousCorrectionDataPreparer
    data_keys = {
        'nl_key': 'NL',
        'fol_key': 'FOL'
    } if translation_task else {
        'nl_key': 'NL',
        'pred_fol_key': 'Pred FOL',
        'comment_key': 'Suggestion',
        'correct_fol_key': 'Correct FOL'
    }

    if continuous_correction:
        assert not translation_task, 'continuous_correction mode only works for correction task'
        data_keys['prev_correct_key'] = 'Prev Correction'

    data_preparer = DataPreparer(
        prompt_template_path,
        tokenizer,
        train_on_inputs,
        cutoff_len
    )
    prepare_input = partial(
        data_preparer.prepare_input,
        **data_keys
    )

    # load data
    train_data, val_data = prepare_dataset(data_path, val_data_path, prepare_input, val_size, data_keys)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if all_exists(val_data_path) else "no",
            save_strategy="steps",
            eval_steps=eval_steps if all_exists(val_data_path) else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if all_exists(val_data_path) else False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    old_collator = trainer.data_collator
    trainer.data_collator = lambda data: dict(old_collator(data))

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
