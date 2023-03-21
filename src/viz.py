import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dist import prepare_scatter


def draw_cormat_lines(
    odorset=None,
    thinwidth=0.1,
    thickwidth=0.5,
    thincolor=1,
    thickcolor=1,
    offset=0,
    scalar=1,
    ax=None,
):
    odorsetSpacing = {
        "global": [],
        "clustered": np.array([4, 7, 11, 14, 18]),
        "tiled": np.array([5, 11, 16]),
    }
    minorSpacing = np.arange(22)
    majorSpacing = odorsetSpacing[odorset]
    for line in minorSpacing:
        if line in majorSpacing:
            ax.axvline(
                line * scalar + offset, color=str(thickcolor), linewidth=thickwidth
            )
            ax.axhline(
                line * scalar + offset, color=str(thickcolor), linewidth=thickwidth
            )
        else:
            ax.axvline(
                line * scalar + offset, color=str(thincolor), linewidth=thinwidth
            )
            ax.axhline(
                line * scalar + offset, color=str(thincolor), linewidth=thinwidth
            )


def heatmap_args():
    heatmap_args = {
        "chem": {
            "vmin": 0,
            "vmax": 1.5,
            "square": True,
            "cmap": "inferno_r",
            "cbar": False,
        },
        "neural": {
            "vmin": 0.2,
            "vmax": 1.1,
            "square": True,
            "cmap": "rocket",
            "cbar": False,
        },
    }
    return heatmap_args


def plot_corrs(
    chem_corrs,
    neural_corrs,
    layers=["l2", "l3"],
    groups=["global", "clustered", "tiled"],
):
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(8, 8),
        gridspec_kw={
            "width_ratios": [1, 0.25, 1, 1],
            "hspace": 0.2,
            "wspace": 0,
        },
    )
    im_args = heatmap_args()
    for ax, (odor_set, v) in zip(axes[:, 0], chem_corrs.items()):
        sns.heatmap(v, xticklabels=[], yticklabels=[], ax=ax, **im_args["chem"])
        ax.set_ylabel(odor_set.title())
    for ax in axes[:, 1]:
        ax.axis("off")

    for ax_row, odor_set in zip(axes[:, 2:], groups):
        for ax, layer in zip(ax_row, layers):
            neural_key = f"{odor_set}_{layer}"
            sns.heatmap(
                neural_corrs[neural_key],
                xticklabels=[],
                yticklabels=[],
                ax=ax,
                **im_args["neural"],
            )
            draw_cormat_lines(
                odorset=odor_set,
                ax=ax,
            )

    axes[0, 0].set_title("Descriptor odor space")
    axes[0, 2].set_title("L2")
    axes[0, 3].set_title("L3")
    axes[-1, 0].set_xlabel("Sorted odor ID")
    return fig


def scatter_dist(
    chem_subset,
    neural_dict,
    dist_keys=["Boutons", "tiled_l2", "tiled_l3", "Model", "TeLC L2", "TeLC L3"],
):
    neural_flat_dist, chem_flat = prepare_scatter(neural_dict, chem_subset, dist_keys)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)

    for (
        ax,
        k,
    ) in zip(axes.flatten(), dist_keys):
        ax.scatter(
            neural_flat_dist[k], chem_flat, fc="None", ec="k", s=20, alpha=0.3, lw=1.5
        )
        ax.set_title(k.replace("tiled_l", "PCx L"))
        ax.set_xlim(0, 1.1)

    sns.despine()
    fig.supxlabel(r"Neural distance $(1-r)$")
    fig.supylabel(r"Chemical distance $(1-r)$")
    return fig