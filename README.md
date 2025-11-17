### This repository contains supporting code for the work ``Understanding Efficiency vs Precision Tradeoffs in Parallel Sampling with MDMs: A Quantitative Study''


### To work with the codebase, please create a conda envorinment and install dependencies, as such:

```bash
git clone 
conda create --name mdm_samples python==3.10
conda activate mdm_samples
cd mdm-samples
pip install -r requirements.txt
```

### Dataset:
Samples will be provided here in a jsonl file.

### Current directory structure:
.
├── artifacts
├── clean.sh
├── data
├── LICENSE
├── README.md
├── requirements.txt
├── results
│   └── baseline.txt
├── scripts
│   ├── baseline_margin.sh
│   ├── eb_sampler.sh
│   ├── fast_dllm.sh
│   ├── get_likeli_ratio.py
│   └── pc_sampler.sh
└── src
    ├── __init__.py
    ├── dev_inference.py
    └── inference.py

### Code references:
https://github.com/NEUIR/PC-Sampler

https://github.com/smarter-vlm/smarter
