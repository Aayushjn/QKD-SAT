from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata


def draw_3d_scatter(dataframe: pd.DataFrame, window_title: str):
    X, Y = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    Z = dataframe["optimal_parts"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, Z, marker="o", c=group["optimal_parts"], cmap=cm.coolwarm)
    ax.invert_xaxis()
    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    ax.set_zlabel("parts")
    fig.colorbar(scatter, shrink=0.5, aspect=5)
    for row in group.itertuples():
        ax.text(x=row.bins[0], y=row.bins[1], z=row.optimal_parts, s=str(row.optimal_parts))

    plt.get_current_fig_manager().set_window_title(window_title)


def draw_3d_surface(dataframe: pd.DataFrame, window_title: str):
    X, Y = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    Z = np.array(dataframe["optimal_parts"])
    points = np.array([X, Y]).T

    x_grid, y_grid = np.mgrid[0 : X.max() : 80j, 0 : Y.max() : 80j]
    z_grid = griddata(points, Z, (x_grid, y_grid), method="cubic")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.summer, linewidth=0, antialiased=True)
    ax.invert_xaxis()
    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    ax.set_zlabel("parts")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    for row in group.itertuples():
        ax.text(x=row.bins[0], y=row.bins[1], z=row.optimal_parts, s=str(row.optimal_parts))

    plt.get_current_fig_manager().set_window_title(window_title)


def draw_heatmap(dataframe: pd.DataFrame, window_title: str):
    X, Y = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    Z = np.array(dataframe["optimal_parts"]).reshape(np.full(2, int(sqrt(len(X)))))

    fig, ax = plt.subplots()
    im = ax.imshow(Z, cmap=cm.summer, interpolation="gaussian", vmin=Z.min(), vmax=Z.max(), origin="lower")

    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    # ax.set_xticks(np.arange(X.max() + 1), labels=X.unique())
    # ax.set_yticks(np.arange(Y.max() + 1), labels=Y.unique())
    # plt.setp(ax.get_xticklabels(), ha="right")

    for i in range(X.max() + 1):
        for j in range(Y.max() + 1):
            ax.text(i, j, Z[i, j], ha="center", va="center")

    fig.colorbar(im, shrink=0.5, aspect=5)

    plt.get_current_fig_manager().set_window_title(window_title)


df = pd.read_csv("results/both.csv").sort_values(by=["graph_size", "bins"])
df["bins"] = df["bins"].apply(lambda x: eval(x))
group_df = df.groupby("graph_size")

for n, group in group_df:
    draw_3d_surface(group, f"Graph Size: {n} nodes")

plt.show()
