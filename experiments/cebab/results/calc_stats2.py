import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    "./results/cmr_results2.csv",
]

plt.rcParams["font.family"] = "Times New Roman"

FONT_SIZE_TITLE = 16
FONT_SIZE_LABELS = 18
FONT_SIZE_TICKS = 16
SHADING_COLOR = 'skyblue'
LINE_COLOR = 'darkblue'
LINE_WIDTH = 2
SHADING_ALPHA = 0.5
lower_limit = 0.75
upper_limit = 1.00
# lower_limit = None
# upper_limit = None
RULE_LIMIT = 28

df = pd.concat([pd.read_csv(f) for f in files])

# get cols "seed", "task_test_acc", "model"
df = df[["seed", "task_test_acc", "model", "n_rules"]]

# compute sdev
df = df.groupby(["n_rules"]).agg(
    task_test_acc_mean=("task_test_acc", np.mean),
    task_test_acc_sdev=("task_test_acc", np.std),
).reset_index()

df_summary = df.sort_values(by="n_rules", ascending=True)

df_summary = df_summary[df_summary["n_rules"] < RULE_LIMIT]

print(df)

plt.figure(figsize=(10, 6))

# Plot mean accuracy as a line
plt.plot(df_summary['n_rules'], df_summary['task_test_acc_mean'], 
         color=LINE_COLOR, linewidth=LINE_WIDTH, marker='o', label='Mean Accuracy')

# Plot standard deviation as shaded area
plt.fill_between(df_summary['n_rules'], 
                 df_summary['task_test_acc_mean'] - df_summary['task_test_acc_sdev'],
                 df_summary['task_test_acc_mean'] + df_summary['task_test_acc_sdev'],
                 color=SHADING_COLOR, alpha=SHADING_ALPHA, label='Standard Deviation')

# Adding labels and title with custom font sizes
plt.xlabel('Number of Rules', fontsize=FONT_SIZE_LABELS)
plt.ylabel('Task Test Accuracy', fontsize=FONT_SIZE_LABELS)

# Customizing ticks
plt.xticks(ticks=df_summary['n_rules'], labels=df_summary['n_rules'], fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

if lower_limit is not None and upper_limit is not None:
    plt.ylim(bottom=lower_limit, top=upper_limit)
elif lower_limit is not None:
    plt.ylim(bottom=lower_limit)
elif upper_limit is not None:
    plt.ylim(top=upper_limit)

# Adding grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=FONT_SIZE_TICKS)

# Reduce whitespace
plt.tight_layout(pad=0.5)

# Show plot
plt.savefig('./new2.pdf')


