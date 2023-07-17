import json
import os
from os.path import join as joinpath
from transformers import LlamaTokenizer
from utils import all_exists, any_exists
from typing import Optional, Dict


class Prompter(object):
    def __init__(self, template_folder_path: str):
        self.template_dict = {}
        for fn in os.listdir(template_folder_path):
            template_name = fn.split('.json')[0]
            with open(joinpath(template_folder_path, fn), 'r') as f:
                self.template_dict[template_name] = json.load(f)

    def generate_prompt(
            self,
            template_name,
            input_kwargs: Optional[Dict[str, str]] = None,
            aux_input_kwargs: Optional[Dict[str, str]] = None,
            output_kwargs: Optional[Dict[str, str]] = None
    ):
        """
            Generate prompt by filling the template; supports generating either input_prompt or output_prompt or both
        """
        assert any_exists(input_kwargs, output_kwargs), 'at least give me one kwargs'

        input_str, output_str = '', ''
        if all_exists(input_kwargs):
            input_str = self.template_dict[template_name]['input_template'].format(**input_kwargs)
        if all_exists(aux_input_kwargs):
            aux_input_str = self.template_dict[template_name]['aux_input_template'].format(**aux_input_kwargs)
            input_str = input_str + aux_input_str
        if all_exists(output_kwargs):
            output_str = self.template_dict[template_name]['output_template'].format(**output_kwargs)
        return input_str, output_str

    def get_response(self, template_name, full_str: str):
        """
            split the output str from the full str and then further split it with separators defined
            in the template['output_splits'] sequentially

            Returns:
                full_response_str: the full model's response str
                resp_parts: a list of response parts specified by the template['output_splits']
        """

        # TODO need a better version of this
        splits = self.template_dict[template_name]['output_splits']
        first_seperator = splits[0]
        full_response_str = (first_seperator if first_seperator in full_str else '') \
                            + full_str.split(first_seperator)[-1]

        first_seq_inds, resp_parts = [], []
        for seq in splits:
            if seq in full_response_str:
                ind = full_response_str.index(seq)
                if (len(first_seq_inds) == 0) or (all_exists(first_seq_inds[-1]) and (ind > first_seq_inds[-1])):
                    first_seq_inds.append(ind)
                    continue
            first_seq_inds.append(None) 
        first_seq_inds = first_seq_inds + [len(full_response_str)]
        for ind, start in enumerate(first_seq_inds[:-1]):
            s = None if start is None else start + len(splits[ind]) + 1
            end = first_seq_inds[ind + 1]
            if all_exists(s, end) and (s <= end):
                resp_parts.append(full_response_str[s:end])
            else:
                resp_parts.append(None)
        return full_response_str, resp_parts


class DataPreparer:

    template_name: str = None

    def __init__(
            self,
            template_folder_path: str,
            tokenizer: LlamaTokenizer,
            train_on_inputs: bool,
            cutoff_len: int
    ):
        self.prompter = Prompter(template_folder_path)
        self.tokenizer = tokenizer

        self.train_on_inputs = train_on_inputs
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt, add_eos_token=True, eval_mode=False, return_tensors=None, **kwargs):
        
        assert (eval_mode and all_exists(return_tensors)) \
               or ((not eval_mode) and (return_tensors is None)), \
               'either use eval mode with return_tensors or not use it and not return_tensors'
        
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=return_tensors,
        )
        
        # if eval mode we just need the input_ids
        if eval_mode:
            return result
        
        if (                
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def make_prompts(self, data_point, **kwargs):
        raise NotImplementedError

    def prepare_input(self, data_point, eval_mode=False, return_tensors=None, **kwargs):
        
        assert (eval_mode and all_exists(return_tensors)) \
               or ((not eval_mode) and (return_tensors is None)), \
               'either use eval mode with return_tensors or not use it and not return_tensors'

        input_prompt, output_prompt = self.make_prompts(data_point, **kwargs)
        
        full_prompt = input_prompt + output_prompt
        
        tokenized_full_prompt = self.tokenize(
            full_prompt, 
            eval_mode=eval_mode,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # for eval mode, input_ids is tensor and we don't need the label modifications
        if eval_mode:
            return tokenized_full_prompt

        if not self.train_on_inputs:
            tokenized_user_prompt = self.tokenize(input_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]

        return tokenized_full_prompt

    def get_input_output_len(self, data_point, **kwargs):
        input_prompt, output_prompt = self.make_prompts(data_point, **kwargs)
        # +1 for the additional eos token during training
        return len(self.tokenizer(input_prompt)['input_ids']) + 1, \
            len(self.tokenizer(output_prompt)['input_ids']) + 1

    def get_response(self, output: str):
        return self.prompter.get_response(self.template_name, output)


class TranslationDataPreparer(DataPreparer):

    template_name = 'translate_prompt_template'

    def make_prompts(
            self,
            data_point,
            nl_key: Optional[str] = None,
            fol_key: Optional[str] = None,
            **kwargs
    ):
        """
            generate prompt for translation task; supports generating either input_prompt or output_prompt or both

            Returns:
                input_prompt: input prompt or '' if nl_key not given
                output_prompt: output prompt or '' if fol_key not given
        """
        assert any_exists(nl_key, fol_key), 'at least give me one key'

        input_prompt, output_prompt = self.prompter.generate_prompt(
            template_name=self.template_name,
            input_kwargs={
                'nl': data_point[nl_key]
            } if all_exists(nl_key) else None,
            output_kwargs={
                'fol': data_point[fol_key]
            } if all_exists(fol_key) else None
        )

        return input_prompt, output_prompt


class CorrectionDataPreparer(DataPreparer):

    template_name = 'correct_prompt_template'

    def make_prompts(
            self,
            data_point,
            nl_key: Optional[str] = None,
            pred_fol_key: Optional[str] = None,
            comment_key: Optional[str] = None,
            correct_fol_key: Optional[str] = None,
            **kwargs
    ):
        """
            generate prompt for correction task; supports generating either input_prompt or output_prompt or both

            Returns:
                input_prompt: input prompt or '' if nl_key+pred_fol_key not given
                output_prompt: output prompt or '' if comment_key+correct_fol_key not given
        """
        assert all_exists(nl_key, pred_fol_key) or all_exists(comment_key, correct_fol_key), \
            'either give me nl_key+pred_fol_key or comment_key+correct_fol_key or both'

        input_prompt, output_prompt = self.prompter.generate_prompt(
            template_name=self.template_name,
            input_kwargs={
                'nl': data_point[nl_key],
                'fol': data_point[pred_fol_key],
            } if all_exists(nl_key, pred_fol_key) else None,
            output_kwargs={
                'comments': data_point[comment_key],
                'fol': data_point[correct_fol_key]
            } if all_exists(comment_key, correct_fol_key) else None
        )

        return input_prompt, output_prompt


class ContinuousCorrectionDataPreparer(DataPreparer):

    template_name = 'continuous_correct_prompt_template'

    def make_prompts(
            self,
            data_point,
            nl_key: Optional[str] = None,
            pred_fol_key: Optional[str] = None,
            prev_correct_key: Optional[str] = None,
            comment_key: Optional[str] = None,
            correct_fol_key: Optional[str] = None,
            **kwargs
    ):
        """
            generate prompt for continuous correction task; supports generating either input_prompt or
            output_prompt or both

            Returns:
                input_prompt: input prompt if nl_key+pred_fol_key are given, and input+aux prompt if prev_correct_key
                is given, otherwise '' if none of nl_key+pred_fol_key are given
                output_prompt: output prompt or '' if comment_key+correct_fol_key not given
        """
        assert all_exists(nl_key, pred_fol_key) or all_exists(comment_key, correct_fol_key), \
            'either give me nl_key+pred_fol_key or comment_key+correct_fol_key or both'

        input_prompt, output_prompt = self.prompter.generate_prompt(
            template_name=self.template_name,
            input_kwargs={
                'nl': data_point[nl_key],
                'fol': data_point[pred_fol_key],
            } if all_exists(nl_key, pred_fol_key) else None,
            aux_input_kwargs={
                'prev_correct': data_point[prev_correct_key]
            } if (all_exists(prev_correct_key) and prev_correct_key in data_point and all_exists(data_point[prev_correct_key])) else None,
            output_kwargs={
                'comments': data_point[comment_key],
                'fol': data_point[correct_fol_key]
            } if all_exists(comment_key, correct_fol_key) else None
        )

        return input_prompt, output_prompt