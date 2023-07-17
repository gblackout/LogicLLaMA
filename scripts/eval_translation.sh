# eval direct translation
python run_eval.py \
    --base_model "path/to/base/model" \
    --peft_path "yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0" \
    --prompt_template_path "data/prompt_templates" \
    --run_name "naive_translate" \
    --translation_task \
    --data_path "path/to/dataset" \
    --save_path "logs/direct_translate_test_evaluation.json" \
    --data_keys '{"nl_key": "NL"}'