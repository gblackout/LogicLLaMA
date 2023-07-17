import json

import fire
import openai
from utils import Prompter, wrap_function_with_timeout, all_exists, make_parent_dirs
from typing import Optional, Dict, List, Union, Callable
from utils import all_exists
import os
from functools import partial
from tqdm import tqdm


folio_5_shot = \
"""

Here are some examples you can refer to:

### NL: 
All citizens of Lawton Park use the zip code 98199.
### Comments:
N/A
### FOL: 
∀x (Citizenof(x, lawtonPark) → Usezipcode(x, number98199))

### NL: 
People either regularly drink coffee or joke about being addicted to caffeine.
### Comments:
N/A
### FOL: 
∀x (Drinks(x) ⊕ Jokes(x))

### NL: 
Museum of Modern Art (MoMA) is a museum if NYC. 
### Comments:
N/A
### FOL: 
Museum(museumofModernArt) ∧ InNYC(museumofModernArt)

### NL: 
Ghosts do not exist.
### Comments:
N/A
### FOL: 
∀x (¬Ghost(x))

### NL: 
Some American radio personalities are also music supervisors. 
### Comments:
N/A
### FOL: 
∃x (American(x) ∧ MusicSupervisor(x) ∧ RadioPersonality(x))

### NL: 
Holding companies hold several companies.
### Comments:
N/A
### FOL: 
∀x ∃y (HoldingCompany(x) → Company(y) ∧ Holds(x, y))

---
"""

logicnli_5_shot = \
"""

Here are some examples you can refer to:

### NL:
If someone is entire, then he is not serious, and vice versa. 
### Comments:
N/A
### FOL:
∃x entire(x) ↔ ¬serious(x)

### NL: 
If there is at least one people who is both not excited and not timid, then Jonathan is elderly.
### Comments:
N/A
### FOL:
∀x (¬excited(x) ∧ ¬timid(x)) → elderly(Jonathan)

### NL: 
Someone who is eithor not fresh or entire is always not serious.
### Comments:
N/A
### FOL:
∀x (¬concerned(x) ∨ fresh(x)) → entire(John)

### NL: 
If Nathalie is not blue, then Collier is entire.
### Comments:
N/A
### FOL:
¬blue(Nathalie) → entire(Collier)

### NL: 
Someone is courteous and not elderly if and only if he is not excited and not various.
### Comments:
N/A
### FOL:
∃x (courteous(x) ∧ ¬elderly(x)) ↔ (¬excited(x) ∧ ¬various(x))

---
"""

translate_system_prompt_5_shot = \
"""
You are a helpful translator that translates natural language (NL) statements into first-order 
logic (FOL) rules. You should
1. Generate the FOL rule that ACCURATELY reflect the meaning of the NL statement
2. USE the following logical operators: ⊕ (either or), ∨ (disjunction), ∧ (conjunction), → (implication), ∀ (universal), ∃ (existential), ¬ (negation), ↔ (equivalence)
3. *NEVER USE* the following symbols for FOL: "!", "≠", "%", "="

Generation Format: you SHOULD ALWAYS generate the translated FOL in the following format
\"\"\"
### Comments:
N/A
### FOL:
{your translated FOL}
\"\"\"

---
"""



translate_system_prompt_zero_shot = \
"""
You are a helpful translator that translates natural language (NL) statements into first-order 
logic (FOL) rules. You should
1. Generate the FOL rule that ACCURATELY reflect the meaning of the NL statement
2. USE the following logical operators: ⊕ (either or), ∨ (disjunction), ∧ (conjunction), → (implication), ∀ (universal), ∃ (existential), ¬ (negation), ↔ (equivalence)
3. *NEVER USE* the following symbols for FOL: "!", "≠", "%", "="

Generation Format: you SHOULD ALWAYS generate the translated FOL in the following format
\"\"\"
### Comments:
N/A
### FOL:
{your translated FOL}
\"\"\"

---
"""


fix_prompt = \
"""
---

Below is one suggestion on how to modify this rule to match the meaning of NL, you SHOULD EITHER:
    1. Fully accept it and change the rule, if you think it is a good suggestion.
    2. Partially accept it and change whatever you think is needed without fully following the suggestion.
    3. Or reject it and do not change the rule, if you think no change is needed.
     
In either case, generate the new FOL in the following format:
\"\"\"
### Comments:
{your comments}
### FOL:
{your new FOL}
\"\"\"

---

Suggestion:

---
"""


class GPTTranslationRequestManager:

    model_gpt4: str = 'gpt-4'
    model_gpt35: str = 'gpt-3.5-turbo'

    def __init__(self, api_key: str):
        """
            Args:
                api_key: either the key string or the path to the key file
        """
        if os.path.isfile(api_key):
            with open(api_key, 'r') as f:
                openai.api_key = f.read().strip()
        else:
            openai.api_key = api_key

    def default_request(
            self,
            input_prompt: str,
            system_prompt: str,
            model: str,
            resp_split_func: Optional[Callable] = None,
            tqdm: Optional[Callable] = None
    ):
        logger = tqdm.write if all_exists(tqdm) else print
        assert model == GPTTranslationRequestManager.model_gpt35 or model == GPTTranslationRequestManager.model_gpt4
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ]
            )
            resp_str = response['choices'][0]['message']['content']
        except:
            logger('something wrong with the request')
            return None

        if all_exists(resp_split_func):
            return resp_split_func(resp_str)

        return resp_str

    def translate_dataset(
            self,
            dataset: Union[str, Dict, List],
            resp_key: str,
            timeout: int = 10,
            resp_split_func: Optional[Callable] = None,
            n_retry: int = 3,
            tqdm: Optional[Callable] = None,
            verbose: bool = False,
            save_path: str = None,
            model: str = 'gpt-3.5-turbo',
            zero_shot: bool = False,
            few_shot_src: Optional[str] = 'folio',
            save_every_nrequests: int = 10,
            src: Optional[str] = None,
    ):

        request_with_timeout = wrap_function_with_timeout(self.default_request, timeout)
        logger = tqdm.write if all_exists(tqdm) else print
        if zero_shot:
            translate_system_prompt = translate_system_prompt_zero_shot
        else:
            assert all_exists(few_shot_src)
            if few_shot_src == 'folio':
                few_shot_example = folio_5_shot
            elif few_shot_src == 'logicnli':
                few_shot_example = logicnli_5_shot
            else:
                raise ValueError(few_shot_src)
            translate_system_prompt = translate_system_prompt_5_shot + few_shot_example

        if isinstance(dataset, str):
            assert os.path.isfile(dataset) and dataset.endswith('json')
            with open(dataset, 'r') as f:
                dataset = json.load(f)

        if isinstance(dataset, Dict):
            assert 'data' in dataset, 'unknown format'
            dataset = dataset['data']

        assert isinstance(dataset, List)

        assert all_exists(save_path)
        make_parent_dirs(save_path)

        pbar = tqdm(dataset, leave=False) if all_exists(tqdm) else None

        update_bar = pbar.update if all_exists(tqdm) else lambda: None

        for ind, entry in enumerate(dataset):
            src_is_valid = (src is None) or (all_exists(src) and entry['src'] == src)
            resp_exists = (resp_key in entry) and all_exists(entry[resp_key])
            should_request = src_is_valid and (not resp_exists)

            if not should_request:
                update_bar()
                continue

            resp = None
            for _ in range(n_retry):
                prompt = f"### NL:\n{entry['NL']}\n"
                resp = request_with_timeout(prompt, translate_system_prompt, model, resp_split_func, tqdm)
                if all_exists(resp):
                    break

            if resp is None:
                logger(f'sample {ind} no response')

            entry[resp_key] = resp[1][1] if all_exists(resp) else None # put the parsed response here
            entry[resp_key + '_full response'] = resp # also keep the orignal response here

            if verbose:
                logger('NL: {0}\nGT FOL: {1}\nGPT FOL:{2}\n---\n'.format(
                    entry['NL'],
                    entry['FOL'] if 'FOL' in entry else None,
                    resp[1][1] if all_exists(resp) else None
                ))

            if ind % save_every_nrequests == 0:
                with open(save_path, 'w') as f:
                    json.dump(dataset, f)

            update_bar()

        with open(save_path, 'w') as f:
            json.dump(dataset, f)


def gpt_translation(
    prompt_path='data/prompt_templates',
    **kwargs
):
    prompter = Prompter(prompt_path)
    resp_split_func = lambda full_str: prompter.get_response('translate_prompt_template', full_str)
    manager = GPTTranslationRequestManager(kwargs['api_key'])
    del kwargs['api_key']

    manager.translate_dataset(
        resp_split_func=resp_split_func,
        tqdm=tqdm,
        **kwargs
    )


if __name__ == '__main__':
    fire.Fire(gpt_translation)