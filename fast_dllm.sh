python eval.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-1.5' \
    --device 'cuda:0' \
    --gen_length 256 \
    --steps 256 \
    --block_length 32 \
    --mode fast_dllm \
    --thread 0.9 \
    --data_path ../data/humaneval20.jsonl \
    --result_path ../results/humaneval_fast_dllm

python ../utils/judge_python_code.py \
    --folder_path ../results/humaneval_fast_dllm \
    --output_path ../results/humaneval_fast_dllm.txt
