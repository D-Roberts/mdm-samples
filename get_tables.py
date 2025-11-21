import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

with open("results/all_metrics.json", "r") as f:
    data = json.load(f)

# print(data.keys())

# df = pd.DataFrame(data["method_margin_steps4_genlen64_blockl32"]["method_metrics"])
# print(df)

d = defaultdict(list)
# print(data.keys())
for id, ablation in data.items():
    # print(id)
    d["run"].append(id)
    # print(ablation.keys())
    for k in ablation.keys():
        # print(k, ablation[k])
        if k not in {"method_metrics", "lambd", "alpha", "gamma", "threashold"}:
            d[k].append(ablation[k])

    for k, v in data[id]["method_metrics"].items():
        # print(k)
        # print("val is ", v)
        if k == "accuracy":
            d[k].append(round(v))
        elif k == "seconds_per_20eg":
            d["secs_sample"].append(round(v / 2))
        else:
            # print(f"this should have k v", v.items())
            for k1, v1 in v.items():
                newlabel = k + k1
                d[newlabel].append(round(v1, 4))

# add some calculated metrics
d["tok_per_sec"] = [j / i for j, i in zip(d["gen_length"], d["secs_sample"])]

# print(f"d is {d}")
newdf = pd.DataFrame(d)

df_new_initial = newdf.drop("run", axis=1)
print(df_new_initial.columns.tolist())

reorder_columns = [
    "method",
    "steps",
    "block_length",
    "gen_length",
    "entropypoint_est",
    "entropysample_std",
    "nllpoint_est",
    "nllsample_std",
    "perplexpoint_est",
    "perplexsample_std",
    "secs_sample",
    "accuracy",
    "tok_per_sec",
]

# print(df_new)
df_new = df_new_initial[reorder_columns]

# split table
cols_to_drop = ["secs_sample", "accuracy", "tok_per_sec"]
copied_df_deep = df_new.copy()
copied_df_deep = copied_df_deep.drop(cols_to_drop, axis=1)

# then drop from initial
cols1_drop = [
    "entropypoint_est",
    "entropysample_std",
    "nllpoint_est",
    "nllsample_std",
    "perplexpoint_est",
    "perplexsample_std",
]
initial_df_drop = df_new.drop(cols1_drop, axis=1)

# Generate the LaTeX table as a string
latex_table = copied_df_deep.to_latex(index=False, float_format="%.2f")
latex_table1 = initial_df_drop.to_latex(index=False, float_format="%.2f")

# # Save the LaTeX code to a file
# with open("output.tex", "w") as f:
#     f.write(latex_table)


# with open("output1.tex", "w") as f:
#     f.write(latex_table1)
