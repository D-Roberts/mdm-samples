import random
import os
import json
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


# current_script_path = os.path.abspath(__file__)
# scripts_dir = os.path.dirname(current_script_path)
# project_root = os.path.dirname(scripts_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "1"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


def load_dataset(data_path, task):
    data_json = load_json_or_jsonl(data_path)
    dataset = []
    for key in data_json.keys():
        dataset.append(data_json[key])
    return dataset


def load_json_or_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                result = {}
                current_key = 1
                for item in data:
                    result[current_key] = item
                    current_key += 1
                return result
            else:
                return data
        except json.JSONDecodeError:
            pass
        result = {}
        f.seek(0)
        current_key = 1
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    for item in parsed:
                        result[current_key] = item
                        current_key += 1
                else:
                    result[current_key] = parsed
                    current_key += 1
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing error on line {line_num}: {e}")
    return result


def humaneval_prompt(func):  # prompt for humaneval
    prompt = f"""Role: You are a professional Python coding assistant
Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
{func}"""
    return prompt


def query_extract(input, task):
    """the task is humaneval"""
    return humaneval_prompt(input["prompt"])


def eval_humaneval(results, dataset, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for index in range(len(results)):
        answer = results[index]
        answer = answer.replace("```python", "\n").replace("```", "\n")
        answer = (
            dataset[index]["prompt"].replace(
                dataset[index]["entry_point"], dataset[index]["entry_point"] + "_prompt"
            )
            + answer
        )
        results[index] = answer

    for index, answer in enumerate(results):
        # print(f"canonical solution is {dataset[index]['canonical_solution']}\n")

        # print(f"while the answer was {answer}\n")

        code_path = f"{result_dir}/{index + 1}.py"
        with open(code_path, "w", encoding="utf-8") as file:
            file.write(answer + "\n")
            file.write(dataset[index]["test"] + "\n")
            file.write('if __name__ == "__main__":\n')
            file.write(f'    check({dataset[index]["entry_point"]})')


def evaluate(task, results, dataset, result_path, args):
    eval_humaneval(results, dataset, result_path)


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
    corpus_name,
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
        from generate import generate_with_margin

        out, entropy = generate_with_margin(
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
        from generate import generate_with_pc_sampler

        out = generate_with_pc_sampler(
            model,
            prompt,
            steps,
            gen_length,
            block_length,
            lambd,
            alpha,
            corpus_name,
            temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    elif mode == "eb_sampler":
        from generate import generate_with_eb_sampler

        out = generate_with_eb_sampler(
            model, prompt, gamma, gen_length, temperature, cfg_scale=0.0
        )
    elif mode == "fast_dllm":
        from generate import generate_with_fast_dllm

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
    return answer, entropy


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
    corpus_name = args.corpus_name
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
    entropies = []

    for input in tqdm(dataset):
        answer, entropy = generate(
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
            corpus_name,
            thread,
            gamma,
            num_remask_tokens,
        )
        results.append(answer)
        entropies.append(entropy)

    evaluate(task, results, dataset, result_path, args)
    print(f"entropies were {entropies} in number {len(entropies)}")

    print("----------------- Done -------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="humaneval")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="margin")
    parser.add_argument("--lambd", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument(
        "--corpus_name",
        type=str,
        default="data/corpus/reference_corpus_llada.json",
    )
    parser.add_argument("--thread", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--num_remask_tokens", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="data/humaneval.jsonl")
    parser.add_argument("--result_path", type=str, default="results/humaneval_results")
    args = parser.parse_args()
    main(args)
