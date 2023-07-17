# eval naive correction on gpt response, the response should be key-value pairs generated from running
# gpt_translation.sh
python run_eval.py \
    --base_model "path/to/base/model" \
    --peft_path "yuan-yang/LogicLLaMA-7b-naive-correction-delta-v0" \
    --prompt_template_path "data/prompt_templates" \
    --run_name "naive_correct" \
    --translation_task=False \
    --data_path "path/to/dataset/with/gpt_response" \
    --save_path "logs/naive_correct_test_evaluation.json" \
    --data_keys '{"nl_key": "NL", "pred_fol_key": "Pred FOL"}'