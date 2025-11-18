# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "---------------------------Fast-dLLM---------------------------"

echo "---------------------------Eval HumanEval---------------------------"

python adaptive_inf.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-1.5' \
    --device 'cuda:0' \
    --gen_length 128 \
    --steps 64 \
    --block_length 32 \
    --mode fast_dllm \
    --thread 0.9 \
    --data_path data/humaneval20.jsonl \
    --result_path results/humaneval_fast_dllm

# python judge_python_code.py \
#     --folder_path results/humaneval_fast_dllm \
#     --output_path results/humaneval_fast_dllm.txt
