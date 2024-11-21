import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import make_interp_spline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vary", required=True, choices=["curiosity", "collaboration"], help="Which variable to vary for plotting"
)
args = parser.parse_args()

csv_file = Path.cwd().joinpath("results").joinpath("results.csv")
df = pd.read_csv(csv_file).sort_values(by=["graph_size", "bins"])
df["bins"] = df["bins"].apply(lambda x: eval(x))  # Convert string to tuple

df["curiosity"] = df["bins"].apply(lambda x: x[0])
df["collaboration"] = df["bins"].apply(lambda x: x[1])

curiosity_groups = df.groupby(["graph_size", "curiosity", "collaboration"])["optimal_parts"].mean().reset_index()
collaboration_groups = (
    df.groupby(["graph_size", "collaboration", "curiosity"])["optimal_parts"].mean().reset_index()
)

unique_collaborations = df["collaboration"].unique()
unique_curiosities = df["curiosity"].unique()
unique_graph_sizes = df["graph_size"].unique()

plt.rcParams.update({"font.size": 18})


def draw_2d_scatter(
    vary_data: npt.NDArray[np.float64],
    primary_data: pd.DataFrame,
    lookup_key: str,
    xkey: str,
    label_prefix: str,
    xlabel: str,
    title: str,
):
    fig = plt.figure(figsize=(10, 6))
    bin_size = 1.0 / vary_data.shape[0]
    for data_point in vary_data:
        subset = primary_data[primary_data[lookup_key] == data_point]

        x_data = subset[xkey].map(lambda x: round(x * bin_size, 2))
        xnew = np.linspace(x_data.min(), x_data.max(), 100)
        spline = make_interp_spline(x_data, subset["optimal_parts"], k=3, check_finite=False)
        range_start = round(data_point * bin_size, 2)
        range_end = round(range_start + bin_size, 2)
        plt.plot(xnew, spline(xnew), label=f"[{range_start}, {range_end})")

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    xticks = np.arange(0, 1, bin_size)
    plt.xticks(xticks, np.vectorize(lambda x: f"[{round(x, 2)}, {round(x + bin_size, 2)})")(xticks))
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel("Shares", labelpad=10)
    plt.legend(title=label_prefix, loc="upper right")
    plt.grid(True)

    fig.savefig(Path.cwd().joinpath("charts").joinpath(f"{args.vary}_{title}.png"))


for graph_size in unique_graph_sizes:
    filtered_curiosity_groups = curiosity_groups[curiosity_groups["graph_size"] == graph_size]
    filtered_collaboration_groups = collaboration_groups[collaboration_groups["graph_size"] == graph_size]

    if args.vary == "curiosity":
        draw_2d_scatter(
            unique_collaborations,
            filtered_curiosity_groups,
            "collaboration",
            "curiosity",
            "Collaboration",
            "Curiosity",
            f"Curiosity vs. Number of Parts for Different Collaboration Values (Graph Size {graph_size})",
        )
    else:
        draw_2d_scatter(
            unique_curiosities,
            filtered_collaboration_groups,
            "curiosity",
            "collaboration",
            "Curiosity",
            "Collaboration",
            f"Collaboration vs. Number of Parts for Different Curiosity Values (Graph Size {graph_size})",
        )

plt.show()
