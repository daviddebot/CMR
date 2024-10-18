import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    "./results/cmr_results1_6.csv",
]

plt.rcParams["font.family"] = "Times New Roman"

# TASK

df = pd.concat([pd.read_csv(f) for f in files])

# get cols "seed", "task_test_acc", "model"
df = df[["seed", "task_test_acc", "model", "n_rules", "use_DT"]]

# # compute sdev
# df = df.groupby(["n_rules", "use_DT"]).agg(
#     task_test_acc_mean=("task_test_acc", np.mean),
#     task_test_acc_sdev=("task_test_acc", np.std),
# ).reset_index()

# df = df.sort_values(by="task_test_acc_mean", ascending=False)

# print(df)


def get_model_name(row):
    if row['model'] == 'DT':
        return 'DT'
    elif not row['use_DT']:
        return 'CMR (n=15)'
    else:
        return f'DT → CMR (n={row["n_rules"]})'
    
df['model_name'] = df.apply(get_model_name, axis=1)

grouped_df = df.groupby('model_name').agg(
    task_test_acc_mean=('task_test_acc', np.mean),
    task_test_acc_sdev=('task_test_acc', np.std)
).reset_index()

# cmr_result = grouped_df.loc[grouped_df['model_name'] == 'CMR (n=15)', 'task_test_acc_mean'].values
# assert len(cmr_result) > 0
# cmr_result = cmr_result[0]

# dt_result = grouped_df.loc[grouped_df['model_name'] == 'DT', 'task_test_acc_mean'].values
# assert len(dt_result) > 0
# dt_result = dt_result[0]

plt.figure(figsize=(10, 6))
plt.bar(
    grouped_df['model_name'],
    grouped_df['task_test_acc_mean'],
    yerr=grouped_df['task_test_acc_sdev'],
    capsize=5,
    alpha=0.7,
    color='skyblue'
)

label_font = 18
ticks_font = 16
lower_limit = 0.8  # 
lower_limit = 0.7
# lower_limit = None

# if cmr_result is not None:
#     plt.axhline(y=cmr_result, color='red', linestyle='--', linewidth=2, label='CMR Result')
# if dt_result is not None:
#     plt.axhline(y=dt_result, color='green', linestyle='--', linewidth=2, label='DT Result')

plt.xlabel('Model', fontsize=label_font)
plt.ylabel('Task Test Accuracy', fontsize=label_font)
# plt.title('Task Test Accuracy with Standard Deviation by Model')
# plt.xticks(rotation=45, ha='right', fontsize=ticks_font)
plt.xticks(fontsize=ticks_font)
plt.yticks(fontsize=ticks_font)
if lower_limit is not None:
    plt.ylim(bottom=lower_limit)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(pad=0.5)
plt.savefig(f"./new_c{files[0][-5]}.pdf")



