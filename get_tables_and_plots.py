import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

# Plots
df_new = df_new_initial[reorder_columns]
print(df_new)

# filter for a setting
df_new_plt = df_new[df_new["gen_length"] == 64][df_new["block_length"] == 32][
    df_new["steps"] == 32
][df_new["method"].isin({"eb_sampler", "fast_dllm", "pc_sampler"})]
print(df_new_plt.columns.tolist())

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
new_ticks = np.arange(3, 14, step=1)


axes[0].scatter(
    df_new_plt["tok_per_sec"],
    df_new_plt["accuracy"],
    cmap="viridis",
    s=100,
    alpha=1.0,
    c=df_new_plt["method"].map(
        {"eb_sampler": "red", "fast_dllm": "blue", "pc_sampler": "green"}
    ),
)

axes[0].set_title("Efficiency vs Precision (num. decoded tokens=64, ceteris paribus)")
axes[0].set_xlabel("Tokens per Second (higher is better)")
axes[0].set_ylabel("Accuracy (%) (higher is better)")
axes[0].set_xticks(new_ticks)


df_new_plt1 = df_new[df_new["gen_length"] == 128][df_new["block_length"] == 32][
    df_new["steps"] == 32
][df_new["method"].isin({"eb_sampler", "fast_dllm", "pc_sampler"})]

axes[1].scatter(
    df_new_plt1["tok_per_sec"],
    df_new_plt1["accuracy"],
    cmap="viridis",
    s=100,
    alpha=1.0,
    c=df_new_plt1["method"].map(
        {"eb_sampler": "red", "fast_dllm": "blue", "pc_sampler": "green"}
    ),
)

axes[1].set_title("Efficiency vs Precision (num. decoded tokens=128, ceteris paribus)")
axes[1].set_xlabel("Tokens per Second (higher is better)")
axes[1].set_ylabel("Accuracy (%) (higher is better)")
axes[1].set_xticks(new_ticks)


# Create a legend
handles = []
labels = []
for method, color in {
    "eb_sampler": "red",
    "fast_dllm": "blue",
    "pc_sampler": "green",
}.items():
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=method,
            markerfacecolor=color,
            markersize=10,
        )
    )
    labels.append(method)
plt.legend(handles, labels, title="Method")
# plt.grid(True)
plt.show()

# # split table
# cols_to_drop = ["secs_sample", "accuracy", "tok_per_sec"]
# copied_df_deep = df_new.copy()
# copied_df_deep = copied_df_deep.drop(cols_to_drop, axis=1)

# # then drop from initial
# cols1_drop = [
#     "entropypoint_est",
#     "entropysample_std",
#     "nllpoint_est",
#     "nllsample_std",
#     "perplexpoint_est",
#     "perplexsample_std",
# ]
# initial_df_drop = df_new.drop(cols1_drop, axis=1)

# # Generate the LaTeX table as a string
# latex_table = copied_df_deep.to_latex(index=False, float_format="%.2f")
# latex_table1 = initial_df_drop.to_latex(index=False, float_format="%.2f")
# print(f"latex_table: {latex_table}")

# # Save the LaTeX code to a file
# with open("output.tex", "w") as f:
#     f.write(latex_table)


# with open("output1.tex", "w") as f:
#     f.write(latex_table1)
