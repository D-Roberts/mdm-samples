 This repository contains supporting code for the work ``Understanding Efficiency vs Precision Tradeoffs in Parallel Sampling with MDMs: A Quantitative Study''.


### To work with the codebase, please create a conda envorinment and install dependencies, as such:

```bash
git clone 
conda create --name mdm_samples python==3.10
conda activate mdm_samples
cd mdm-samples
pip install -r requirements.txt
```

### Dataset:
Samples from human eval code infill task are in data/humaneval20.jsonl.

### To run ablations and generate the "all_metrics.json" in the working directory:
```
python adaptive_inf.py
```

### Code to get latex tables (with json, defaultdict, and pandas)
```
get_tables.py
```

### Code references:
https://github.com/NEUIR/PC-Sampler

https://github.com/D-Roberts/LLaDA

https://github.com/smarter-vlm/smarter
