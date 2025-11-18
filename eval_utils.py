import os, json, re, csv, sys
from pathlib import Path
from typing import Dict, List, Optional

current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
        code_path = f"{result_dir}/{index + 1}.py"
        with open(code_path, "w", encoding="utf-8") as file:
            file.write(answer + "\n")
            file.write(dataset[index]["test"] + "\n")
            file.write('if __name__ == "__main__":\n')
            file.write(f'    check({dataset[index]["entry_point"]})')


def evaluate(task, results, dataset, result_path, args):
    eval_humaneval(results, dataset, result_path)
