export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "---------------------------PC-Sampler---------------------------"

python adaptive_inf.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-1.5' \
    --device 'cuda:0' \
    --gen_length 128 \
    --steps 64 \
    --block_length 128 \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 10 \
    --baseline_name /home/ubuntu/mdm-samples/data/baseline/reference_corpus_llada.json \
    --data_path data/humaneval20.jsonl \
    --result_path results/humaneval_margin

python judge_python_code.py \
    --folder_path results/humaneval_pc_sampler \
    --output_path results/humaneval_pc_sampler.txt
