import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--vary", required=True, choices=["curiosity", "collaboration"], help="Which variable to vary for plotting"
)
args = parser.parse_args()

# Load the appropriate CSV file based on the --vary argument
csv_file = f"results/{args.vary}.csv"
df = pd.read_csv(csv_file).sort_values(by=["graph_size", "bins"])
df["bins"] = df["bins"].apply(lambda x: eval(x))  # Convert string to tuple

# Separate the bins into curiosity and collaboration columns
df["curiosity"] = df["bins"].apply(lambda x: x[0])
df["collaboration"] = df["bins"].apply(lambda x: x[1])

# Group by curiosity and collaboration and calculate the average number of parts for each group
curiosity_groups = df.groupby(["curiosity", "collaboration"])["optimal_parts"].mean().reset_index()
collaboration_groups = df.groupby(["collaboration", "curiosity"])["optimal_parts"].mean().reset_index()

# Unique curiosity and collaboration values for plotting
unique_collaborations = df["collaboration"].unique()
unique_curiosities = df["curiosity"].unique()

# Conditional plotting based on the --vary argument
if args.vary == "curiosity" or args.vary == "both":
    # Plot Graph 1: Curiosity vs. Number of Parts with fixed Collaboration
    plt.figure(figsize=(10, 6))
    for collaboration in unique_collaborations:
        subset = curiosity_groups[curiosity_groups["collaboration"] == collaboration]
        plt.plot(subset["curiosity"], subset["optimal_parts"], label=f"Collaboration {collaboration}")
    y_min, y_max = 0, 10
    y_step = 2.5
    plt.yticks(np.arange(y_min, y_max, y_step))
    plt.xlabel("Curiosity (Bins)")
    plt.ylabel("Number of Parts")
    plt.title("Curiosity vs. Number of Parts for Different Collaboration Values")
    plt.legend(title="Collaboration", loc="upper right")
    plt.grid(True)
    plt.show()

if args.vary == "collaboration" or args.vary == "both":
    # Plot Graph 2: Collaboration vs. Number of Parts with fixed Curiosity
    plt.figure(figsize=(10, 6))
    for curiosity in unique_curiosities:
        subset = collaboration_groups[collaboration_groups["curiosity"] == curiosity]
        plt.plot(subset["collaboration"], subset["optimal_parts"], label=f"Curiosity {curiosity}")
    y_min, y_max = 0, 10
    y_step = 2.5
    plt.yticks(np.arange(y_min, y_max, y_step))
    plt.xlabel("Collaboration (Bins)")
    plt.ylabel("Number of Parts")
    plt.title("Collaboration vs. Number of Parts for Different Curiosity Values")
    plt.legend(title="Curiosity", loc="upper right")
    plt.grid(True)
    plt.show()
