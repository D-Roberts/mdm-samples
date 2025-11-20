import os, sys
import numpy as np
import torch.nn.functional as F
import torch

# this is for pc_sampler weights
CORPUS = None


def entropy_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError(
            "Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]"
        )
    epsilon = 1e-12
    probs_safe = probabilities.clone() + epsilon
    entropy = torch.sum(probabilities.clone() * torch.log(probs_safe), dim=-1)
    return entropy


def nll_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError(
            "Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]"
        )
    epsilon = 1e-12
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    probs_safe = top1_probs.clone() + epsilon
    print(f"what is prob safe {top1_probs.shape[1]}")
    ll = torch.sum(torch.log(probs_safe), dim=-1) / probs_safe.shape[1]
    return -ll


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# for baseline
def margin_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError(
            "Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]"
        )
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    confidence = top1_probs - top2_probs
    return confidence


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def load_corpus(model, corpus_name):
    global CORPUS
    if CORPUS is None:
        from adaptive_inf import load_json_or_jsonl

        corpus_dict = load_json_or_jsonl(corpus_name)
        token_num_ = corpus_dict["num_token"]
        corpus_dict = corpus_dict["corpus_dict"]
        del_keys = []
        for key in corpus_dict.keys():
            del_keys.append(key)
        for key in del_keys:
            corpus_dict[int(key)] = corpus_dict[key]
        for key in del_keys:
            del corpus_dict[key]
        for key in corpus_dict.keys():
            corpus_dict[key] = corpus_dict[key] / token_num_
        CORPUS = torch.full(
            (126464,), 1 / token_num_, device=model.device, dtype=torch.float32
        )
        keys = torch.tensor(
            list(corpus_dict.keys()), device=model.device, dtype=torch.long
        )
        values = torch.tensor(
            list(corpus_dict.values()), device=model.device, dtype=torch.float32
        )
        CORPUS.scatter_(0, keys, values)
    else:
        CORPUS = CORPUS.to(model.device)


@torch.no_grad()
def generate_with_margin(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    return_order=False,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            x0_p = margin_function(p[:, prompt.shape[1] :])

            # metrics last step only
            if (num_block == num_blocks - 1) and (i == steps - 1):
                entropy = (
                    -entropy_function(p[:, prompt.shape[1] :]).sum() / block_length
                )

                sequence_len = logits.shape[1]
                # print(
                #     f"logits shape {logits.shape} and masked logits shape {logits[mask_index].shape}"
                # ) # batch 1, seqlen, vocab

                nll = nll_function(
                    p[:, prompt.shape[1] :][mask_index[:, prompt.shape[1] :]]
                )
                print(f"nll {nll}")

            confidence = torch.where(mask_index[:, prompt.shape[1] :], x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index + prompt.shape[1]] = True
                if return_order:
                    if num_block + 1 not in orders:
                        orders[num_block + 1] = []
                    orders[num_block + 1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]

    return x, entropy.detach().cpu().item()


def pc_sampler_function(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    lambda_val: float,
    alpha: float,
    bg_freq_tensor: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape != token_ids.shape:
        raise f"probabilities.shape: {probabilities.shape}, token_ids.shape: {token_ids.shape} must be equal"

    device = probabilities.device
    sequence_len = probabilities.shape[1]
    f_bg_tensor = bg_freq_tensor[token_ids]
    epsilon = 1e-9
    cross_entropy_scores = -probabilities * torch.log(f_bg_tensor + epsilon)
    cross_entropy_scores = torch.clamp(cross_entropy_scores, max=alpha)

    # print(f"cross entropy scores in pc sampler fn {cross_entropy_scores} ")
    # print(
    #     f"so then perplexity min and max {min(torch.exp(cross_entropy_scores)[0])} and {max(torch.exp(cross_entropy_scores)[0])}"
    # )

    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = torch.exp(-lambda_val * positions)
    final_scores = positional_bias * cross_entropy_scores

    return final_scores


@torch.no_grad()
def generate_with_pc_sampler(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    lambd=1,
    alpha=1,
    corpus_name="corpus.json",
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    return_order=False,
):
    # this is the weighing of tokens by frequency to account for bias of confidence toward some tokens
    global CORPUS
    if CORPUS is None:
        load_corpus(model, corpus_name)
    if return_order:
        orders = {}

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)

            x0_p = pc_sampler_function(
                probabilities=x0_p[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ],
                token_ids=x0[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ],
                lambda_val=lambd,
                alpha=alpha,
                bg_freq_tensor=CORPUS,
            )

            if i == steps - 1:
                entropy = (
                    -entropy_function(p[:, prompt.shape[1] :]).sum() / block_length
                )

            confidence = torch.where(
                mask_index[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ],
                x0_p,
                -np.inf,
            )
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[
                    j, select_index + prompt.shape[1] + num_block * block_length
                ] = True
                if return_order:
                    if num_block + 1 not in orders:
                        orders[num_block + 1] = []
                    orders[num_block + 1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x, entropy.detach().cpu().item()


@torch.no_grad()
def generate_with_eb_sampler(
    model,
    prompt,
    gamma=0.1,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    mask_id=126336,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    while (x == mask_id).any():
        mask_index = x == mask_id

        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        masked_logits = logits[mask_index]

        # get entropy
        p = F.softmax(logits, dim=-1)
        entropy = -entropy_function(p[:, prompt.shape[1] :]).sum() / gen_length

        err_proxy = torch.distributions.Categorical(logits=masked_logits).entropy()

        masked_token_indices = mask_index.nonzero(as_tuple=True)[1]
        sorted_err_indices = torch.argsort(err_proxy)
        sorted_indices = masked_token_indices[sorted_err_indices]

        sorted_entropies = err_proxy[sorted_err_indices]

        acc_entropy = torch.cumsum(sorted_entropies, dim=0)
        cummax_entropy = torch.cummax(sorted_entropies, dim=0).values

        k = (acc_entropy - cummax_entropy <= gamma).sum()

        num_masks_available = len(sorted_indices)
        k = torch.clamp(k, min=1, max=num_masks_available)

        indices_to_unmask = sorted_indices[:k]

        x[0, indices_to_unmask] = predicted_tokens[0, indices_to_unmask]

    return x, entropy.detach().cpu().item()


# for fast dllm
def get_transfer_index(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


# for fast dllm
def get_transfer_index_dynamic(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


@torch.no_grad()
def generate_with_fast_dllm(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits

            p = F.softmax(logits, dim=-1)
            entropy = -entropy_function(p[:, prompt.shape[1] :]).sum() / block_length
            # print(f"entropy in fast dllm {entropy}")
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor
                )
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (
                x[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break
    # print(f"why not enough val to unpack {x} and {entropy.detach().cpu().item()}")
    return x, entropy.detach().cpu().item()
