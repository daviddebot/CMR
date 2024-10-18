import pandas as pd
import numpy as np

files = [
    "./results/mnist_base/results_2024-10-25_15-50-54.csv",
    "./results/mnist_base/comp_results_2024-10-25_10-35-06.csv",
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

input("Press enter for next results")

# INTERVENTIONS

files = [
    "./results/mnist_base/int_results_2024-10-21_13-44-37.csv",
]

df = pd.concat([pd.read_csv(f) for f in files])

# create new col which is task_test_acc_after_int-task_test_acc_before_int
df["task_test_acc_diff"] = df["task_test_acc_after_int"] - df["task_test_acc_before_int"]

# group by model and compute mean and sdev
df = df.groupby(["model"]).agg(
    task_test_acc_diff_mean=("task_test_acc_diff", np.mean),
    task_test_acc_diff_sdev=("task_test_acc_diff", np.std),
).reset_index()

print(df)

