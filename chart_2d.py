import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import make_interp_spline

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--vary", required=True, choices=["curiosity", "collaboration"], help="Which variable to vary for plotting"
)
args = parser.parse_args()

# Load the appropriate CSV file based on the --vary argument
csv_file = Path.cwd().joinpath("results").joinpath("both.csv")
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


def draw_2d_scatter(
    vary_data: npt.NDArray[np.float64],
    primary_data: pd.DataFrame,
    lookup_key: str,
    xkey: str,
    label_prefix: str,
    xlabel: str,
    title: str,
):
    plt.figure(figsize=(10, 6))
    for data_point in vary_data:
        subset = primary_data[primary_data[lookup_key] == data_point]
        xnew = np.linspace(subset[xkey].min(), subset[xkey].max(), 100)
        cs = make_interp_spline(subset[xkey], subset["optimal_parts"], k=2, check_finite=False)
        plt.plot(xnew, cs(xnew), label=f"{label_prefix} {data_point}")

    # plt.xticks(np.arange(0, 10, 1.0))
    # plt.yticks(np.arange(y_min, y_max, y_step))
    plt.xlabel(f"{xlabel} (Bins)")
    plt.ylabel("Number of Parts")
    plt.title(title)
    plt.legend(title=label_prefix, loc="upper right")
    plt.grid(True)
    plt.show()


if args.vary == "curiosity":
    draw_2d_scatter(
        unique_collaborations,
        curiosity_groups,
        "collaboration",
        "curiosity",
        "Collaboration",
        "Curiosity",
        "Curiosity vs. Number of Parts for Different Collaboration Values",
    )
else:
    draw_2d_scatter(
        unique_curiosities,
        collaboration_groups,
        "curiosity",
        "collaboration",
        "Curiosity",
        "Collaboration",
        "Collaboration vs. Number of Parts for Different Curiosity Values",
    )

# Conditional plotting based on the --vary argument
# if args.vary == "curiosity" or args.vary == "both":
#     # Plot Graph 1: Curiosity vs. Number of Parts with fixed Collaboration
#     plt.figure(figsize=(10, 6))
#     for collaboration in unique_collaborations:
#         subset = curiosity_groups[curiosity_groups["collaboration"] == collaboration]
#         plt.plot(subset["curiosity"], subset["optimal_parts"], label=f"Collaboration {collaboration}")
#     y_min, y_max = 0, 10
#     y_step = 2.5
#     plt.yticks(np.arange(y_min, y_max, y_step))
#     plt.xlabel("Curiosity (Bins)")
#     plt.ylabel("Number of Parts")
#     plt.title("Curiosity vs. Number of Parts for Different Collaboration Values")
#     plt.legend(title="Collaboration", loc="upper right")
#     plt.grid(True)
#     plt.show()

# if args.vary == "collaboration" or args.vary == "both":
#     # Plot Graph 2: Collaboration vs. Number of Parts with fixed Curiosity
#     plt.figure(figsize=(10, 6))
#     for curiosity in unique_curiosities:
#         subset = collaboration_groups[collaboration_groups["curiosity"] == curiosity]
#         plt.plot(subset["collaboration"], subset["optimal_parts"], label=f"Curiosity {curiosity}")
#     y_min, y_max = 0, 10
#     y_step = 2.5
#     plt.yticks(np.arange(y_min, y_max, y_step))
#     plt.xlabel("Collaboration (Bins)")
#     plt.ylabel("Number of Parts")
#     plt.title("Collaboration vs. Number of Parts for Different Curiosity Values")
#     plt.legend(title="Curiosity", loc="upper right")
#     plt.grid(True)
#     plt.show()
