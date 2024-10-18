import pandas as pd
import numpy as np

files = [
    "./results/cebab_base/cmr_results1.csv",
    "./results/cebab_base/cbm_competitors1.csv",
]

# TASK

df = pd.concat([pd.read_csv(f) for f in files])

# get cols "seed", "task_test_acc", "model"
df = df[["seed", "task_test_acc", "model"]]

# compute sdev
df = df.groupby(["model"]).agg(
    task_test_acc_mean=("task_test_acc", np.mean),
    task_test_acc_sdev=("task_test_acc", np.std),
).reset_index()

df = df.sort_values(by="task_test_acc_mean", ascending=False)

print(df)

# CONCEPT

df = pd.concat([pd.read_csv(f) for f in files])

# get cols "seed", "task_test_acc", "model"
df = df[["seed", "concepts_test_acc", "model"]]

# compute sdev
df = df.groupby(["model"]).agg(
    task_test_acc_mean=("concepts_test_acc", np.mean),
    task_test_acc_sdev=("concepts_test_acc", np.std),
).reset_index()

print(df)

