import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

allowed_seeds = [1, 2, 3]

used_n_c = [1, 12, 24, 37]

plt.rcParams["font.family"] = "Times New Roman"

unallowed_model_names = [
    "XGBoost (ground truth)",
    "XGBoost (input embedding)",
    "Decision Tree (ground truth)",
    "StandardDCR",
    "CEMDeep",
    "DCR",
    "CBM+linear",
]

def name_mapper(name):
    if name == "XGBoost2":
        return "CBM+XG"
    if name == "DT2":
        return "CBM+DT"
    if name == "CEMDeept1":
        return "CEM"
    if name == "StandardDCRt1":
        return "DCR"
    if name == "HardCBMDeep":
        return "CBM+MLP"
    if name == "HardCBMLin":
        return "CBM+linear"
    if name == "GDCR":
        return "CMR"
    if name == "DNN":
        return "Black box"
    else:
        return name


files = [
    "./results/celeba_base/cmr_results_2024-10-27_21-00-07.csv",
    "./results/celeba_base/comp_results_2024-10-25_10-35-16.csv"

]

# TASK

# open each file
df = pd.concat([pd.read_csv(file) for file in files])

# get columns "seed", "n_concepts",  "task_test_acc", "model"
df = df[["seed", "n_concepts", "task_test_acc", "model"]]

# map model
df["model"] = df["model"].apply(name_mapper)

# filter seeds
df = df[df["seed"].isin(allowed_seeds)]

# compute sdev
df["task_test_acc_sdev"] = df.groupby(["n_concepts", "model"])["task_test_acc"].transform(np.std)

# average over seed
df = df.groupby(["n_concepts", "model"]).mean().reset_index()

# filter unallowed model names
df = df[~df["model"].isin(unallowed_model_names)]
df = df[df["n_concepts"].isin(used_n_c)]

# plot
fig, ax = plt.subplots()
ax.yaxis.get_major_locator().set_params(nbins=6)
for model in df["model"].unique():
    df_model = df[df["model"] == model]
    ax.plot(df_model["n_concepts"], df_model["task_test_acc"], label=model, marker='o')

# increase
font_size = 25
plt.xlabel("Number of Concepts", fontsize=font_size)
plt.ylabel("Task Test Accuracy", fontsize=font_size)
plt.xticks(df["n_concepts"].unique(), fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(loc='lower right', fontsize=13)
plt.tight_layout()
# plt.show()
plt.savefig('./celeba.pdf')

# print rows
print(df)

input("Press enter for next results")

# CONCEPTS

unallowed_model_names.append("DNN")

# open each file
df = pd.concat([pd.read_csv(file) for file in files])

# get columns "seed", "n_concepts",  "task_test_acc", "model"
df = df[["seed", "n_concepts", "concepts_test_acc", "model"]]

# map model
df["model"] = df["model"].apply(name_mapper)

# filter seeds
df = df[df["seed"].isin(allowed_seeds)]

# compute sdev
df["concepts_test_acc_sdev"] = df.groupby(["n_concepts", "model"])["concepts_test_acc"].transform(np.std)

# average over seed
df = df.groupby(["n_concepts", "model"]).mean().reset_index()

# filter unallowed model names
df = df[~df["model"].isin(unallowed_model_names)]

# plot
fig, ax = plt.subplots()
for model in df["model"].unique():
    df_model = df[df["model"] == model]
    ax.plot(df_model["n_concepts"], df_model["concepts_test_acc"], label=model, marker='o')
plt.xlabel("Number of Concepts")
plt.ylabel("Concepts Test Accuracy")
plt.legend()
# plt.show()

# pandas print all cols
pd.set_option('display.max_columns', 100)

# set inf line width
pd.set_option('display.width', 1000)

# print rows
print(df)




