from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata


def draw_3d_scatter(dataframe: pd.DataFrame, window_title: str):
    x_data, y_data = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    z_data = dataframe["optimal_parts"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(x_data, y_data, z_data, marker="o", c=group["optimal_parts"], cmap=cm.coolwarm)
    ax.invert_xaxis()
    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    ax.set_zlabel("parts")
    fig.colorbar(scatter, shrink=0.5, aspect=5)
    for row in group.itertuples():
        ax.text(x=row.bins[0], y=row.bins[1], z=row.optimal_parts, s=str(row.optimal_parts))

    plt.get_current_fig_manager().set_window_title(window_title)


def draw_3d_surface(dataframe: pd.DataFrame, window_title: str):
    plt.rcParams.update({"font.size": 18})
    x_data, y_data = dataframe["bins"].apply(lambda x: x[0]) / 10, dataframe["bins"].apply(lambda x: x[1]) / 10
    z_data = np.array(dataframe["optimal_parts"])
    points = np.array([x_data, y_data]).T

    x_grid, y_grid = np.mgrid[0 : x_data.max() : 80j, 0 : y_data.max() : 80j]
    z_grid = griddata(points, z_data, (x_grid, y_grid), method="cubic")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 15))
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.invert_xaxis()
    ax.set_xlabel("curiosity", labelpad=10)
    ax.set_ylabel("collaboration", labelpad=10)
    ax.set_zlabel("shares")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # for row in group.itertuples():
    #     ax.text(x=row.bins[0], y=row.bins[1], z=row.optimal_parts, s=str(row.optimal_parts))

    plt.get_current_fig_manager().set_window_title(window_title)


def draw_heatmap(dataframe: pd.DataFrame, window_title: str):
    x_data, y_data = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    z_data = np.array(dataframe["optimal_parts"]).reshape(np.full(2, int(sqrt(len(x_data)))))

    fig, ax = plt.subplots()
    im = ax.imshow(
        z_data, cmap=cm.summer, interpolation="gaussian", vmin=z_data.min(), vmax=z_data.max(), origin="lower"
    )

    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")

    for i in range(x_data.max() + 1):
        for j in range(y_data.max() + 1):
            ax.text(i, j, z_data[i, j], ha="center", va="center")

    fig.colorbar(im, shrink=0.5, aspect=5)

    plt.get_current_fig_manager().set_window_title(window_title)


df = pd.read_csv("results/results.csv").sort_values(by=["graph_size", "bins"])
df["bins"] = df["bins"].apply(lambda x: eval(x))
group_df = df.groupby("graph_size")

for n, group in group_df:
    draw_3d_surface(group, f"Graph Size: {n} nodes")

plt.show()
