import os

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting
from nilearn.image import new_img_like


Name = [
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "MMSE",
    "FAQ",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
]

Gene_num = 0

ATLAS_FILE = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
INFO_FILE = "Schaefer2018_200Parcels_7Networks_order_info.txt"
FEATS_FILE = "feats.txt"
EMBED_FILE = f"Gene_Outcome/Gene_{Gene_num}"
GEN_FILE = f"Gene_Outcome/Gene_{Gene_num}"

for subdir in ("plot", "plot/v1", "plot/v2"):
    os.makedirs(subdir, exist_ok=True)

atlas_img = nib.load(ATLAS_FILE)
atlas_data = atlas_img.get_fdata()
region_label_map = pd.read_csv(INFO_FILE).set_index("feature_name")["region_label"]

with open(FEATS_FILE) as f:
    feats = [line.strip() for line in f]

assert len(feats) == 202, "feats.txt must contain exactly 202 labels."


def _roi_dict_to_img(roi_dict):
    """Convert ROI-level values into a NIfTI image.

    Parameters
    ----------
    roi_dict : dict
        Dictionary mapping ROI feature names to scalar values.

    Returns
    -------
    nib.Nifti1Image
        NIfTI image with values assigned according to atlas labels.
    """
    arr = np.zeros_like(atlas_data, dtype=np.float32)

    for feat, val in roi_dict.items():
        lab = region_label_map.get(feat)

        if lab is not None:
            arr[atlas_data == lab] = val

    return new_img_like(atlas_img, arr)


def make_cmp(R, G, B, N=256):
    """Create a listed colormap from RGB channel ranges.

    Parameters
    ----------
    R : array-like
        Two-element red-channel range.
    G : array-like
        Two-element green-channel range.
    B : array-like
        Two-element blue-channel range.
    N : int, default=256
        Number of colors.

    Returns
    -------
    ListedColormap
        Generated colormap.
    """
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(R[0] / 256, R[1] / 256, N)
    vals[:, 1] = np.linspace(G[0] / 256, G[1] / 256, N)
    vals[:, 2] = np.linspace(B[0] / 256, B[1] / 256, N)

    return ListedColormap(vals)


def make_double_cmp(R1, G1, B1, R2, G2, B2, N=256):
    """Create a two-sided listed colormap from two RGB gradients.

    Parameters
    ----------
    R1, G1, B1 : array-like
        RGB channel ranges for the first half of the colormap.
    R2, G2, B2 : array-like
        RGB channel ranges for the second half of the colormap.
    N : int, default=256
        Number of colors.

    Returns
    -------
    ListedColormap
        Combined colormap.
    """
    top = make_cmp(R1, G1, B1, N=N)
    bottom = make_cmp(R2, G2, B2, N=N)

    new_colors = np.vstack(
        [
            top(np.linspace(0, 1, int(N / 2))),
            bottom(np.linspace(1, 0, int(N / 2))),
        ]
    )

    return ListedColormap(new_colors)


def plot_brain_feature_importances(fi_dict, label=("v1", "q1"), vmin=None, vmax=None):
    """Plot and save brain feature importances on a glass-brain map.

    Parameters
    ----------
    fi_dict : dict
        Dictionary mapping ROI feature names to importance values.
    label : tuple[str, str], default=("v1", "q1")
        Output subdirectory and filename suffix.
    vmin : float, optional
        Lower color-scale limit. If `None`, a symmetric scale is inferred.
    vmax : float, optional
        Upper color-scale limit. If `None`, a symmetric scale is inferred.
    """
    if vmin is None or vmax is None:
        abs_max = max(abs(min(fi_dict.values())), abs(max(fi_dict.values())))

        if abs_max == 0:
            abs_max = 1e-12

        vmin, vmax = -abs_max, abs_max

    RR = 202
    RG = 0
    RB = 32

    BR = 5
    BG = 113
    BB = 176

    W = 256

    img = _roi_dict_to_img(fi_dict)
    my_cmap = make_double_cmp(
        [RR, W],
        [RG, W],
        [RB, W],
        [BR, W],
        [BG, W],
        [BB, W],
    )

    plotting.plot_glass_brain(
        img,
        display_mode="lyrz",
        vmin=vmin,
        vmax=vmax,
        cmap=my_cmap,
        plot_abs=False,
        colorbar=False,
    )

    out = f"plot/{label[0]}/fi_{label[1]}"
    plt.savefig(out + ".svg", bbox_inches="tight")
    plt.close()


def plot_feature_importance_quantile(q: float = 0.95, gene_cols: list | None = None):
    """Plot thresholded brain maps for each outcome row.

    The function loads two feature-importance matrices, applies robust
    column-wise scaling based on the 95th percentile of absolute values, and
    generates one brain map per outcome row for each matrix.

    Parameters
    ----------
    q : float, default=0.95
        Row-wise quantile threshold. This value is computed internally but the
        final thresholding rule currently uses a fixed numerical cutoff.
    gene_cols : list, optional
        Unused parameter kept for API compatibility.
    """
    df = pd.read_csv(EMBED_FILE, index_col=0)
    df.columns = feats

    df_gen = pd.read_csv(GEN_FILE, index_col=0)
    df_gen.columns = feats

    comb = pd.concat([df, df_gen])[feats[2:]]
    scale = comb.abs().quantile(0.95)
    scale[scale == 0] = 1e-12

    def _scale(frame):
        """Apply robust column-wise scaling to ROI columns."""
        return frame[feats[2:]] / scale

    df_s = _scale(df)
    df_gen_s = _scale(df_gen)

    def _plot_rows(frame, prefix):
        """Plot one brain map for each row in a scaled dataframe."""
        for idx, row in frame.iterrows():
            abs_vals = row.abs().values
            nz_vals = abs_vals[abs_vals > 0]

            if nz_vals.size == 0:
                continue

            thr_row = np.quantile(nz_vals, q)
            abs_max = nz_vals.max()

            fi_dict = {
                feature: value if abs(value) >= 1e-6 else 0.0
                for feature, value in row.items()
            }

            plot_brain_feature_importances(
                fi_dict=fi_dict,
                label=(prefix, f"{Name[idx]}"),
                vmin=-abs_max,
                vmax=abs_max,
            )

    _plot_rows(df_s, "v1")
    _plot_rows(df_gen_s, "v2")

    print(
        f"[INFO] Generated thresholded brain maps using q={q:.2f} "
        "after robust column-wise scaling."
    )


def plot_display(ax, roi_dict, vmin, vmax):
    """Plot ROI-level values on a glass-brain display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the plot.
    roi_dict : dict
        Dictionary mapping ROI feature names to values.
    vmin : float
        Lower color-scale limit.
    vmax : float
        Upper color-scale limit.
    """
    RR = 202
    RG = 0
    RB = 32

    BR = 5
    BG = 113
    BB = 176

    W = 256

    img = _roi_dict_to_img(roi_dict)
    my_cmap = make_double_cmp(
        [RR, W],
        [RG, W],
        [RB, W],
        [BR, W],
        [BG, W],
        [BB, W],
    )

    plotting.plot_glass_brain(
        img,
        axes=ax,
        display_mode="lyrz",
        vmin=vmin,
        vmax=vmax,
        cmap=my_cmap,
        plot_abs=False,
        colorbar=False,
    )


def plot_grid_11x2():
    """Plot a two-column grid comparing two feature-importance matrices."""
    if not os.path.exists(GEN_FILE):
        print("Comparison file was not found. Skipping the 11x2 grid.")
        return

    df1 = pd.read_csv(GEN_FILE, index_col=0)
    df1.columns = feats

    df2 = pd.read_csv(EMBED_FILE, index_col=0)
    df2.columns = feats

    rows = min(11, len(df1))
    fig, axes = plt.subplots(rows, 2, figsize=(20, rows * 3))

    for i in range(rows):
        for j, df_src in enumerate((df1, df2)):
            row = df_src.iloc[i][feats[2:]]
            abs_max = row.abs().max() or 1e-12

            plot_display(
                axes[i, j],
                row.to_dict(),
                vmin=-abs_max,
                vmax=abs_max,
            )

    plt.tight_layout()
    plt.savefig("plot/brain_importance_grid.svg", bbox_inches="tight")
    plt.close()


def plot_grid_35x7():
    """Plot feature-importance brain maps by functional network."""
    df = pd.read_csv(EMBED_FILE, index_col=0)
    df.columns = feats

    networks = [
        ("Vis", "Visual"),
        ("SomMot", "SomMot"),
        ("DorsAttn", "DorsAttn"),
        ("SalVentAttn", "Sal/Vent"),
        ("Limbic", "Limbic"),
        ("Cont", "Control"),
        ("Default", "Default"),
    ]

    rows = len(df)
    cols = 7

    fig, axes = plt.subplots(rows, cols, figsize=(70, 40), dpi=50000)

    for r, (_, row) in enumerate(df[feats[2:]].iterrows()):
        abs_max = row.abs().max() or 1e-12

        for c, (kw, _) in enumerate(networks):
            roi_dict = {
                feature: row[feature]
                for feature in feats[2:]
                if kw in feature
            }

            plot_display(
                axes[r, c],
                roi_dict,
                vmin=-abs_max,
                vmax=abs_max,
            )

            if r == 0:
                axes[r, c].set_title(kw, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"plot/brain_importance_{rows}x7.svg", bbox_inches="tight")
    plt.close()


def plot_network_change_over_quantiles():
    """Plot average network-level feature importance across rows."""
    df = pd.read_csv(EMBED_FILE, index_col=0)
    df.columns = feats

    networks = [
        ("Vis", "Visual"),
        ("SomMot", "SomMot"),
        ("DorsAttn", "DorsAttn"),
        ("SalVentAttn", "Sal/Vent"),
        ("Limbic", "Limbic"),
        ("Cont", "Control"),
        ("Default", "Default"),
    ]

    means = {}
    sems = {}

    for kw, name in networks:
        cols = [feature for feature in feats[2:] if kw in feature]
        mat = df[cols].to_numpy()

        means[name] = mat.mean(axis=1)
        sems[name] = mat.std(axis=1) / np.sqrt(mat.shape[1])

    plt.figure(figsize=(8, 6))

    for name in means:
        plt.errorbar(
            df.index,
            means[name],
            yerr=sems[name],
            marker="o",
            capsize=3,
            label=name,
        )

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Mean importance")
    plt.title("Network change across rows")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot/network_importance_trend.svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_feature_importance_quantile()
    # plot_grid_35x7()
    # plot_grid_11x2()
    # plot_network_change_over_quantiles()