import os
import json
import numpy as np
from itertools import combinations, chain
import torch
import random
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "1"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

device = "mps"
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    trust_remote_code=True,
    padding="max_length",
    max_length=16,
    truncation=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "bert-base-uncased", trust_remote_code=True, is_decoder=True
)

model.to(device)
model.eval()


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(
        torch.linspace(
            float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
        )
    ).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat(
        (
            torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
            is_mask,
        ),
        dim=1,
    )
    # print(f"what is is_mask in the forward process {is_mask.shape}") #batch size, sequence len (prompt+ans)
    noisy_batch = torch.where(is_mask, mask_id, batch)
    # print(f"then noisy batch shape {noisy_batch.shape} and val {noisy_batch}")

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    input = batch
    logits = model(input).logits

    if cfg_scale > 0.0:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@torch.no_grad()
def get_log_likelihood(
    model, prompt, answer, mc_num=128, batch_size=2, cfg_scale=0.0, mask_id=126336
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    """
    print(
        f"what is prompt shape {prompt.shape} and answer shape {answer.shape}"
    )  # token ids from the prompt
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    print(
        f"what is seq after concat prompt and answer and then repeat {seq.shape}"
    )  # batch size, prompt+ans
    # tokens of shape #16, 29=16+13 from prompt and answer.

    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)
    print(
        f"the prompt shape would be prompt_index {prompt_index.shape} and val {prompt_index}"
    )

    loss_ = []
    for i in range(mc_num // batch_size):
        print(f"at step i")
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        # p_mask is mask ratio

        print(f"here p_mask {p_mask}")

        mask_index = perturbed_seq == mask_id  # this will be a mask of true and false

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)
        print(
            f"logits[mask_index] shape {seq[mask_index].shape}"
        )  # torch.Size([12]) and logits of shape [12, vocabsize]

        loss = (
            F.cross_entropy(logits[mask_index], seq[mask_index], reduction="none")
            / p_mask[mask_index]
        )
        print(
            f"and p_mask[mask_index] {p_mask[mask_index]} and shape {p_mask[mask_index].shape}"
        )

        print(f"loss is {loss}")
        print(
            f"loss shape before div by pmask {F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none').shape}"
        )

        perplex = torch.exp(loss)

        # print(f"logits[mask_index] values {logits[mask_index][0]}")
        # print(f"seq[mask_index] values {seq[mask_index][0]}")
        # print(f"p_mask[mask_index] shape {p_mask[mask_index]}")

        # print(
        #     f"perplex as exp avg loss per seq {perplex.shape} {perplex[0].item(): .4f}"
        # )

        loss = loss.sum() / batch_size

        loss_.append(loss.item())
        ret = -sum(loss_) / len(loss_)  # this is -NLL averaged over batch
        print(f"what was nll {ret}")
    return ret


def main():
    # device = "cuda"
    # model_name = "GSAI-ML/LLaDA-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # # vocabulary  size 126464;

    # model = AutoModel.from_pretrained(
    #     model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    # ).to(device)

    # model.to(device)
    # model.eval()

    # Model architecture
    # print(model)

    # Likelihood ratio test

    prompt = "Roof shingle removal: A man is sitting on a roof. He"
    answer = " is using wrap to wrap a pair of skis."

    # for i in range(1, 21):
    #     # file_path_ground = f"/Users/dr/research/mdm-samples/results/humaneval_results/ground_truth/{i}.py"
    #     file_path_ground = (
    #         f"/home/ubuntu/mdm-samples/results/humaneval_results/ground_truth/{i}.py"
    #     )
    #     # file_path_answer = (f"/Users/dr/research/mdm-samples/results/humaneval_results/{i}.py")
    #     file_path_answer = (
    #         f"/home/ubuntu/mdm-samples/results/humaneval_pc_sampler/{i}.py"
    #     )

    #     print(f"for case {i} *** ")
    #     try:
    #         with open(file_path_ground, "r") as file:
    #             file_content_ground = file.read()
    #         # print("File content as a string:")
    #         # print(file_content_ground)
    #     except FileNotFoundError:
    #         print(f"Error: The file '{file_path_ground}' was not found.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")

    #     try:
    #         with open(file_path_answer, "r") as file:
    #             file_content_answer = file.read()
    #         # print("File content as a string:")
    #         # print(file_content_answer)
    #     except FileNotFoundError:
    #         print(f"Error: The file '{file_path_answer}' was not found.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")

    ground_tokens = torch.tensor(tokenizer(prompt)["input_ids"]).to(device)
    print(
        f"tokenized ground tokens input ids are of shape {ground_tokens.shape} and value {ground_tokens}"
    )

    #     # set dummy answer the same as ground truth to test this.
    answer_tokens = torch.tensor(tokenizer(answer)["input_ids"]).to(device)
    print(
        f"tokenized ground tokens input ids are of shape {answer_tokens.shape} and value {answer_tokens}"
    )

    ref = get_log_likelihood(model, ground_tokens, answer_tokens, mc_num=4)
    # print(f"the ref log likelihood ******** {ref}")
    # current = get_log_likelihood(model, ground_tokens, answer_tokens, mc_num=64)
    # ratio = current - ref
    # print(f"Log likelihood ratio is: {ratio}")


if __name__ == "__main__":
    main()
