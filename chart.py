from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def draw_3d_scatter(dataframe: pd.DataFrame, window_title: str):
    X, Y = dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1])
    Z = dataframe["optimal_parts"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, Z, marker="o", c=group["optimal_parts"], cmap=cm.cool)
    ax.invert_xaxis()
    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    ax.set_zlabel("parts")
    fig.colorbar(scatter, shrink=0.5, aspect=5)
    for row in group.itertuples():
        ax.text(x=row.bins[0], y=row.bins[1], z=row.optimal_parts, s=str(row.optimal_parts))

    plt.get_current_fig_manager().set_window_title(window_title)


def draw_3d_surface(dataframe: pd.DataFrame, window_title: str):
    X, Y = np.meshgrid(dataframe["bins"].apply(lambda x: x[0]), dataframe["bins"].apply(lambda x: x[1]))
    Z = np.array(
        [dataframe.loc[dataframe["bins"] == (x, y)]["optimal_parts"].values for x, y in zip(np.ravel(X), np.ravel(Y))]
    ).reshape(X.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.cool, linewidth=0, antialiaseds=False)
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
    Z = np.array([dataframe.loc[dataframe["bins"] == (x, y)]["optimal_parts"].values for x, y in zip(X, Y)]).reshape(
        np.full(2, int(sqrt(len(X))))
    )

    fig, ax = plt.subplots()
    im = ax.imshow(Z, cmap=cm.cool, origin="lower")

    ax.set_xlabel("curiosity")
    ax.set_ylabel("collaboration")
    ax.set_xticks(np.arange(X.max() + 1), labels=X.unique())
    ax.set_yticks(np.arange(Y.max() + 1), labels=Y.unique())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(X.max() + 1):
        for j in range(Y.max() + 1):
            ax.text(i, j, Z[i, j], ha="center", va="center")

    fig.colorbar(im, shrink=0.5, aspect=5)

    plt.get_current_fig_manager().set_window_title(window_title)


df = pd.read_csv("results/both.csv")
df["bins"] = df["bins"].apply(lambda x: eval(x))
group_df = df.groupby("graph_size")

for n, group in group_df:
    draw_3d_scatter(group, f"Graph Size: {n} nodes")

plt.show()
