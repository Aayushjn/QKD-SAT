import csv

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/curiosity.csv")

fig, axes = plt.subplots(1, 2)

for n, group in df.groupby("n"):
    axes[0].plot(group["bins"], group["parts"], label=f"n={n}")

# Add labels and title
axes[0].set_xlabel("bins")
axes[0].set_ylabel("parts")
axes[0].set_title("Curiosity")
axes[0].legend()


df = pd.read_csv("results/collaboration.csv")
for n, group in df.groupby("n"):
    axes[1].plot(group["bins"], group["parts"], label=f"Nodes={n}")

# Add labels and title
axes[1].set_xlabel("bins")
axes[1].set_ylabel("parts")
axes[1].set_title("Collaboration")
axes[1].legend()

fig.tight_layout()
# Show the plot
plt.show()
