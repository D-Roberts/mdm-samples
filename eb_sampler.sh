export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "---------------------------EB-Sampler---------------------------"

echo "---------------------------Eval HumanEval---------------------------"

python adaptive_inf.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:0' \
    --gen_length 128 \
    --steps 64 \
    --block_length 128 \
    --mode eb_sampler \
    --gamma 0.01 \
    --data_path data/humaneval20.jsonl \
    --result_path results/humaneval_eb_sampler

python judge_python_code.py \
    --folder_path results/humaneval_eb_sampler \
    --output_path results/humaneval_eb_sampler.txt
