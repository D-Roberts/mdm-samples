### Check reproducibility

- Loglikelihood ratios, for the same random seed (0), on the same input, when comparing two
different sampling number of steps:

    prompt = torch.tensor(tokenizer(prompt)["input_ids"]).to(device)
    answer = torch.tensor(tokenizer(answer)["input_ids"]).to(device)
    ref = get_log_likelihood(model, prompt, answer, mc_num=128)
    current = get_log_likelihood(model, prompt, answer, mc_num=64)
    ratio = current - ref
    print(f"Log likelihood ratio is: {ratio}")

Log likelihood ratio is: -4.389980316162109 
