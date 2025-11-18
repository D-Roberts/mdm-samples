import random
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from utils.eval_utils import load_dataset, evaluate, query_extract

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "1"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


def generate(
    model,
    tokenizer,
    input,
    task,
    steps,
    gen_length,
    block_length,
    temperature,
    mode,
    lambd,
    alpha,
    baseline_name,
    thread,
    gamma,
    num_remask_tokens=10,
):
    # the query depends on the task which in this case is humaneval
    # the model is instruction tuned so we apply a special chat template which gives the model the prompt
    query = query_extract(input, task)
    m = [{"role": "user", "content": query}]
    user_input = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )
    prompt = tokenizer(user_input)["input_ids"]
    prompt = torch.tensor(prompt).to(model.device).unsqueeze(0)

    # the baseline and compared methods
    if mode == "margin":
        from src.generate import generate_with_margin

        out = generate_with_margin(
            model,
            prompt,
            steps,
            gen_length,
            block_length,
            temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    elif mode == "pc_sampler":
        from src.generate import generate_with_pc_sampler

        out = generate_with_pc_sampler(
            model,
            prompt,
            steps,
            gen_length,
            block_length,
            lambd,
            alpha,
            baseline_name,
            temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    elif mode == "eb_sampler":
        from src.generate import generate_with_eb_sampler

        out = generate_with_eb_sampler(
            model, prompt, gamma, gen_length, temperature, cfg_scale=0.0
        )
    elif mode == "fast_dllm":
        from src.generate import generate_with_fast_dllm

        out = generate_with_fast_dllm(
            model,
            prompt,
            steps,
            gen_length,
            block_length,
            temperature,
            remasking="low_confidence",
            threshold=thread,
        )[0]

    answer = tokenizer.batch_decode(
        out[:, prompt.shape[1] :], skip_special_tokens=True
    )[0]
    return answer


def main(args):
    task = args.task
    model_name = args.model_name
    device = args.device
    gen_length = args.gen_length
    steps = args.steps
    block_length = args.block_length
    temperature = args.temperature
    mode = args.mode
    lambd = args.lambd
    alpha = args.alpha
    baseline_name = args.baseline_name
    thread = args.thread
    gamma = args.gamma
    num_remask_tokens = args.num_remask_tokens
    data_path = args.data_path
    result_path = args.result_path

    dataset = load_dataset(data_path, task)

    print("----------------- Load model -------------------")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print("----------------- Start Answering -------------------")

    results = []
    for input in tqdm(dataset):
        answer = generate(
            model,
            tokenizer,
            input,
            task,
            steps,
            gen_length,
            block_length,
            temperature,
            mode,
            lambd,
            alpha,
            baseline_name,
            thread,
            gamma,
            num_remask_tokens,
        )
        results.append(answer)

    evaluate(task, results, dataset, result_path, args)

    print("----------------- Done -------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="humaneval")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="margin")
    parser.add_argument("--lambd", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument(
        "--baseline_name", type=str, default="../data/baseline/reference_corpus.json"
    )
    parser.add_argument("--thread", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--num_remask_tokens", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="./data/humaneval.jsonl")
    parser.add_argument(
        "--result_path", type=str, default="../results/humaneval_results"
    )
    args = parser.parse_args()
    main(args)
