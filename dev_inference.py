import json
import numpy as np
from itertools import combinations, chain
import torch
import random
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer

# load model once at start
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
