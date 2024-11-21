from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import alexandergovern
from scipy.stats import f_oneway
from scipy.stats import kruskal

csv_file = Path.cwd().joinpath("data copy.csv")
df = pd.read_csv(csv_file).sort_values(by=["graph_size", "bins"])
df["bins"] = df["bins"].apply(lambda x: eval(x))
df["curiosity"] = df["bins"].apply(lambda x: x[0])
df["collaboration"] = df["bins"].apply(lambda x: x[1])

curiosities = [df[df["curiosity"] == i]["breaking_probability"].values for i in range(10)]
collaborations = [df[df["collaboration"] == i]["breaking_probability"].values for i in range(10)]

cur_std = np.std(curiosities, axis=1)
col_std = np.std(curiosities, axis=1)

if len(set(cur_std)) == 1 and len(set(col_std)) == 1:
    print("====== One-way ANOVA ======")
    print(f_oneway(*curiosities))
    print(f_oneway(*collaborations))
else:
    print("Skipping One-way ANOVA since variances are not equal")

print("====== Kruskal ======")
print(kruskal(*curiosities))
print(kruskal(*collaborations))

print("====== Alexander-Govern ======")
print(alexandergovern(*curiosities))
print(alexandergovern(*collaborations))
