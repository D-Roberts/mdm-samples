python eval.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-1.5' \
    --device 'cuda:0' \
    --gen_length 128 \
    --steps 64 \
    --block_length 128 \
    --mode margin \
    --data_path ../data/humaneval20.jsonl \
    --result_path ../results/humaneval_margin

python ../src/judge_python_code.py \
    --folder_path ../results/humaneval_margin \
    --output_path ../results/humaneval_margin.txt
