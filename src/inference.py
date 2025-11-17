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

# load model once at start
device = "cuda"
model_name = "GSAI-ML/LLaDA-1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# vocabulary  size 126464;

model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)

model.to(device)
model.eval()
