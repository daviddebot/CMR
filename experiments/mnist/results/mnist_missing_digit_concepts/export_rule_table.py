import os
import pandas as pd
import numpy as np

files = [
    "./results_2024-10-25_22-37-19.csv",
]

tables = []


df = pd.concat([pd.read_csv(f) for f in files])

def clean(rule):
    rule = rule.split(" <- ")
    rule = "y_{" + rule[0][1:] + "}" + " \leftarrow " + rule[1]
    return rule.replace("&", "\\land").replace("~", "\\neg ")

for seed in df["seed"].unique():

    df1 = df[["seed", "learned_and_selected_rules"]]

    all_rules = set()
    for i, row in df1[df1["seed"] == seed].iterrows():
        rules = eval(row["learned_and_selected_rules"])
        for rule in rules:
            all_rules.add(clean(rule))

    # to list and sort
    all_rules = list(all_rules)
    all_rules = sorted(all_rules)

    table = ""
    table += "\\begin{figure}\n\\centering\n\\scriptsize\n"
    # table += f"\\caption{{Rulebook for MNIST+$^*$ (seed {seed}).}}\n"
    # table += f"\\label{{tab:rules_mnist_missing_seed_{seed}}}\n"

    # table += f"\\subfloat[Seed {seed + 1}]{{"
    # table += "\\begin{tabular}{l}\n\\toprule\n"

    # tables.append("")
    # tables[seed-1] += f"\\begin{{table}}\n\\centering\n\\caption{{Rulebook for MNIST+$^*$ (seed {seed}).}}\n\\scriptsize\n"
    # tables[seed-1] += "\\begin{tabular}{p{14cm}}\n\\toprule\n"
    all_rules = sorted(all_rules, key=lambda x: int(x.split(" \leftarrow ")[0][3:].replace('}', '')))
    n_subfloats = 0
    for idx, rule in enumerate(all_rules):
        if idx % 24 == 0:
            n_subfloats += 1
            if idx != 0:
                # table += "\\bottomrule\n\end{tabular}}\n"
                table += "\end{tabular}}\n"
                table += "\\hfill\n"
                if n_subfloats % 5 == 0:
                    table += "\\newline \n"
            table += f"\\subfloat[]{{"
            table += "\\begin{tabular}{l}"
            if idx == 0:
                table += "\\toprule\n"
            else:
                table += "\n"
        table += "$" + rule + "$ \\\ \n"
        if idx+1 != len(all_rules):
            task = int(rule.split(" \leftarrow ")[0][3:].replace('}', ''))
            next_task = int(all_rules[idx+1].split(" \leftarrow ")[0][3:].replace('}', ''))
            if task != next_task:
                table += "\\midrule\n"
        # table += " \midrule \n"
    # table = table[:-12] + "\\ \n"
    table += "\\bottomrule\n\end{tabular}}\n" + f"\label{{tab:rules_mnist_missing_seed_{seed}}}" #+ "\\end{table}\n"
    table += f"\\caption{{Rulebook for MNIST+$^*$ (seed {seed}).}}\n"
    table += "\\end{figure}\n"
    tables.append(table)


with open("./mnist_missing.tex", "w") as f:
    for table in tables:
        f.write(table)
        f.write("\n\n")

