import pandas as pd
import numpy as np

files = [
    "./results/mnist_missing_digit_concepts/results_2024-10-25_22-37-19.csv",
    "./results/comps_mnist_missing_digit_concepts/results_2024-10-25_23-35-34.csv",
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

print(df)

# CONCEPT

df = pd.concat([pd.read_csv(f) for f in files])

# get cols "seed", "task_test_acc", "model"
df = df[["seed", "concepts_test_acc", "model"]]

# compute sdev
df = df.groupby(["model"]).agg(
    concepts_test_acc_mean=("concepts_test_acc", np.mean),
    concepts_test_acc_sdev=("concepts_test_acc", np.std),
).reset_index()

print(df)



