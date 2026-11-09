"""Render the data-independent workflow and architecture figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduced" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#1F4E79",
    "blue": "#0072B2",
    "light_blue": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "pink": "#CC79A7",
    "gray": "#666666",
    "border": "#4D4D4D",
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def add_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    facecolor,
    edgecolor="#4D4D4D",
    fontsize=8.3,
    linewidth=0.9,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )
    return patch


def add_arrow(ax, start, end, color="#4D4D4D", linewidth=1.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=linewidth,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def render_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(
        ax,
        0.16,
        0.82,
        0.68,
        0.12,
        "Women with breast cancer treated with NAC and surgery, 2018-2023\n"
        "n = 105 (pCR, n = 27; non-pCR, n = 78)",
        "#DCE9F5",
        COLORS["navy"],
        fontsize=9.2,
    )
    add_box(ax, 0.19, 0.61, 0.18, 0.13, "Clinical variables\n11 prespecified inputs", "#E8F3FB", COLORS["blue"])
    add_box(ax, 0.41, 0.61, 0.18, 0.13, "DCE-MRI\n6 enhancement phases", "#E6F4FA", COLORS["light_blue"])
    add_box(ax, 0.63, 0.61, 0.18, 0.13, "H&E biopsy WSI\nUNI patch features", "#E7F5F0", COLORS["green"])
    add_box(ax, 0.19, 0.43, 0.18, 0.12, "Nested 5-fold CV\nclinical MLP", "#F8FAFC", COLORS["blue"], fontsize=8.0)
    add_box(ax, 0.41, 0.43, 0.18, 0.12, "Saved out-of-fold\n3D DenseNet-121", "#F8FAFC", COLORS["light_blue"], fontsize=8.0)
    add_box(ax, 0.63, 0.43, 0.18, 0.12, "Saved out-of-fold\nattention MIL", "#F8FAFC", COLORS["green"], fontsize=8.0)
    add_box(
        ax,
        0.24,
        0.235,
        0.52,
        0.11,
        "Fixed equal-weight aggregation of out-of-fold probabilities\n(no weight tuning on evaluation data)",
        "#FCEAF3",
        COLORS["pink"],
        fontsize=8.5,
    )
    add_box(
        ax,
        0.21,
        0.055,
        0.58,
        0.12,
        "Paired bootstrap comparisons and exploratory calibration/DCA\n"
        "Report fold-level variability and pooled out-of-fold AUC with 95% CI",
        "#F0F0F0",
        COLORS["gray"],
        fontsize=8.2,
    )
    for x_value in (0.28, 0.50, 0.72):
        add_arrow(ax, (x_value, 0.82), (x_value, 0.745))
        add_arrow(ax, (x_value, 0.61), (x_value, 0.55))
        add_arrow(ax, (x_value, 0.43), (x_value, 0.345))
    add_arrow(ax, (0.50, 0.235), (0.50, 0.175))
    ax.text(
        0.02,
        0.01,
        "Abbreviations: CV, cross-validation; DCA, decision curve analysis; H&E, "
        "haematoxylin and eosin; MIL, multiple-instance learning; WSI, whole-slide image.",
        fontsize=6.8,
        ha="left",
        va="bottom",
    )
    save_figure(fig, "Figure_1_Analysis_Workflow")


def render_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_clinical = 0.70
    add_box(ax, 0.02, y_clinical, 0.12, 0.14, "11 clinical\nvariables", "#E8F3FB", COLORS["blue"])
    add_box(ax, 0.19, y_clinical, 0.17, 0.14, "MLP\n11 -> 64 -> 32 -> 16", "#F6FBFE", COLORS["blue"], fontsize=8.0)
    add_box(ax, 0.41, y_clinical, 0.12, 0.14, "Clinical\nclassifier", "#FFFFFF", COLORS["blue"])

    y_mri = 0.43
    add_box(ax, 0.02, y_mri, 0.12, 0.14, "DCE-MRI\n6 phases", "#E6F4FA", COLORS["light_blue"])
    add_box(ax, 0.19, y_mri, 0.17, 0.14, "3D DenseNet-121\nfrom scratch", "#F6FBFE", COLORS["light_blue"], fontsize=8.0)
    add_box(ax, 0.41, y_mri, 0.12, 0.14, "512-D\nfeature\nclassifier", "#FFFFFF", COLORS["light_blue"], fontsize=7.9)

    y_pathology = 0.16
    add_box(ax, 0.02, y_pathology, 0.12, 0.14, "H&E biopsy\nWSI", "#E7F5F0", COLORS["green"])
    add_box(ax, 0.19, y_pathology, 0.17, 0.14, "224 px patches\nfrozen UNI -> 1024-D", "#F6FBFE", COLORS["green"], fontsize=7.8)
    add_box(ax, 0.41, y_pathology, 0.12, 0.14, "Gated attention\nMIL -> 1024-D", "#FFFFFF", COLORS["green"], fontsize=7.6)
    for y_value in (y_clinical + 0.07, y_mri + 0.07, y_pathology + 0.07):
        add_arrow(ax, (0.14, y_value), (0.19, y_value))
        add_arrow(ax, (0.36, y_value), (0.41, y_value))
    add_box(
        ax,
        0.61,
        0.20,
        0.18,
        0.70,
        "Per-modality\nlogits / probabilities\n\nFixed average\n\n1 / 3 each\n\n"
        "No tuning on\nevaluation folds",
        "#FCEAF3",
        COLORS["pink"],
        fontsize=8.0,
    )
    add_box(ax, 0.84, 0.42, 0.14, 0.18, "Predicted\npCR\nprobability", "#FFF2CC", COLORS["orange"], fontsize=8.4)
    for y_value in (y_clinical + 0.07, y_mri + 0.07, y_pathology + 0.07):
        add_arrow(ax, (0.53, y_value), (0.61, y_value))
    add_arrow(ax, (0.79, 0.51), (0.84, 0.51))
    ax.text(
        0.02,
        0.045,
        "The clinical branch was re-estimated using nested cross-validation. MRI and "
        "pathology probabilities were reused from saved out-of-fold predictions.\n"
        "The fixed aggregation shown here is the prespecified benchmark analysis, "
        "not a tuned fusion model.",
        fontsize=6.9,
        ha="left",
        va="bottom",
        linespacing=1.35,
    )
    save_figure(fig, "Figure_2_Model_Architecture")


def main() -> None:
    render_workflow()
    render_architecture()
    print(f"Rendered Figures 1-2 to {OUTPUT}")


if __name__ == "__main__":
    main()
