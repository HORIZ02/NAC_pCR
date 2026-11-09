"""Create the extended visualization suite from frozen out-of-fold predictions.

All tabular figures are generated from the deidentified analysis files in the
parent directory. Patient-level source data are written next to each figure so
that every plotted value can be traced without redistributing the original
clinical workbook.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_visualizations"
FIGURE_DIR = OUT / "figures"
SOURCE_DIR = OUT / "figure_source"
TABLE_DIR = OUT / "tables"
for directory in (OUT, FIGURE_DIR, SOURCE_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    ("clinical_11", "Clinical (11 variables)", "Clinical (11)"),
    ("mri", "MRI", "MRI"),
    ("pathology", "Pathology", "Pathology"),
    ("clinical_plus_mri", "Clinical + MRI", "Clinical+MRI"),
    ("clinical_plus_pathology", "Clinical + pathology", "Clinical+Path"),
    ("trimodal_equal_weight_11", "Clinical + MRI + pathology", "Trimodal"),
]
MODEL_COLORS = {
    "clinical_11": "#D62728",
    "mri": "#1F77B4",
    "pathology": "#2CA02C",
    "clinical_plus_mri": "#FF7F0E",
    "clinical_plus_pathology": "#17BECF",
    "trimodal_equal_weight_11": "#9467BD",
}
OUTCOME_COLORS = {0: "#7A7A7A", 1: "#00A087"}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "axes.labelpad": 5,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    }
)

INDEX_ROWS: list[dict[str, object]] = []
ATLAS_PATH = OUT / "VISUALIZATION_ATLAS.pdf"
ATLAS = PdfPages(ATLAS_PATH)
PERMUTATION_SOURCE = SOURCE_DIR / "EV18_Clinical_permutation_importance_source.csv"


def model_table(oof: pd.DataFrame) -> pd.DataFrame:
    return oof[["subject_id", "outer_fold", "pCR"] + [key for key, _, _ in MODEL_ORDER]].copy()


def bootstrap_auc_draws(
    y: np.ndarray,
    probability: np.ndarray,
    folds: np.ndarray,
    n_boot: int = 3000,
    seed: int = 20261103,
) -> np.ndarray:
    """Stratified paired bootstrap, resampling within each outer fold."""
    rng = np.random.default_rng(seed)
    fold_indices = [np.flatnonzero(folds == fold) for fold in np.unique(folds)]
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        draws[index] = roc_auc_score(y[sampled], probability[sampled])
    return draws


def paired_bootstrap_difference(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    folds: np.ndarray,
    n_boot: int = 5000,
    seed: int = 20261103,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    fold_indices = [np.flatnonzero(folds == fold) for fold in np.unique(folds)]
    differences = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        differences[index] = roc_auc_score(y[sampled], probability_a[sampled]) - roc_auc_score(
            y[sampled], probability_b[sampled]
        )
    observed = float(roc_auc_score(y, probability_a) - roc_auc_score(y, probability_b))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_value = float(2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
    return observed, float(lower), float(upper), min(1.0, p_value)


def net_benefit(y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(y))
    result = np.empty_like(thresholds, dtype=float)
    for index, threshold in enumerate(thresholds):
        predicted = probability >= threshold
        tp = float(np.sum((predicted == 1) & (y == 1)))
        fp = float(np.sum((predicted == 1) & (y == 0)))
        result[index] = tp / len(y) - fp / len(y) * threshold / (1 - threshold)
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    figure: plt.Figure,
    figure_id: str,
    title: str,
    source: pd.DataFrame,
    *,
    section: str = "statistical",
    evidence: str = "Frozen OOF descriptive analysis",
    notes: str = "",
) -> None:
    for axis in figure.axes:
        axis.grid(False)
        axis.tick_params(axis="both", which="both", direction="out", length=3, width=0.8, pad=5)
        axis.xaxis.labelpad = 5
        axis.yaxis.labelpad = 5
    stem = f"{figure_id}_{title.replace(' ', '_').replace('/', '_').replace('+', 'plus')}"
    figure_path = FIGURE_DIR / f"{stem}.png"
    source_path = SOURCE_DIR / f"{stem}_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    figure.savefig(figure_path, dpi=320, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(
        FIGURE_DIR / f"{stem}.tiff",
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    ATLAS.savefig(figure, bbox_inches="tight")
    INDEX_ROWS.append(
        {
            "figure_id": figure_id,
            "title": title,
            "status": "complete",
            "section": section,
            "evidence_level": evidence,
            "png": str(figure_path.relative_to(OUT)),
            "source_csv": str(source_path.relative_to(OUT)),
            "notes": notes,
        }
    )
    plt.close(figure)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(ROOT / "analysis_dataset_deidentified.csv")
    oof = pd.read_csv(ROOT / "out_of_fold_predictions.csv")
    per_fold = pd.read_csv(ROOT / "per_fold_metrics.csv")
    reported = json.loads((ROOT / "reported_values.json").read_text(encoding="utf-8"))
    return dataset, oof, per_fold, reported


def figure_pr_curves(oof: pd.DataFrame) -> None:
    y = oof["pCR"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    source_rows = []
    for key, label, _ in MODEL_ORDER:
        p = oof[key].to_numpy()
        precision, recall, thresholds = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=MODEL_COLORS[key], lw=1.7, label=f"{label}  AP={ap:.3f}")
        for r, q, threshold in zip(recall[:-1], precision[:-1], thresholds):
            source_rows.append(
                {"model_key": key, "model_label": label, "recall": r, "precision": q, "threshold": threshold}
            )
    prevalence = float(np.mean(y))
    ax.axhline(prevalence, color="#000000", lw=1.0, ls="--", label=f"No-skill prevalence={prevalence:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (sensitivity)", ylabel="Precision (positive predictive value)")
    ax.grid(color="#E5E5E5", lw=0.6)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout()
    save_figure(fig, "EV01", "Precision recall curves", pd.DataFrame(source_rows))


def figure_risk_waterfall(oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharey=True)
    source_rows = []
    panel_keys = MODEL_ORDER[:3] + [MODEL_ORDER[-1]]
    for panel, (ax, (key, label, short)) in zip("ABCD", zip(axes.flat, panel_keys)):
        frame = oof[["subject_id", "outer_fold", "pCR", key]].sort_values(key).reset_index(drop=True)
        predicted = (frame[key] >= 0.5).astype(int)
        colors = [OUTCOME_COLORS[value] for value in frame["pCR"]]
        ax.bar(np.arange(len(frame)), frame[key], color=colors, width=0.82, linewidth=0)
        wrong = predicted.to_numpy() != frame["pCR"].to_numpy()
        ax.scatter(np.flatnonzero(wrong), frame.loc[wrong, key], marker="x", s=18, color="#B2182B", linewidth=0.9)
        ax.axhline(0.5, color="#000000", lw=0.9, ls="--")
        ax.set_title(short)
        ax.set(xlim=(-1, len(frame)), ylim=(0, 1.02), xlabel="Patients ordered by predicted probability", ylabel="Predicted pCR probability")
        ax.set_xticks([0, 20, 40, 60, 80, 104])
        ax.grid(axis="y", color="#E8E8E8", lw=0.6)
        add_panel_label(ax, panel)
        for row in frame.assign(predicted=predicted, correct=~wrong).to_dict("records"):
            source_rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "subject_id": row["subject_id"],
                    "outer_fold": row["outer_fold"],
                    "pCR": row["pCR"],
                    "probability": row[key],
                    "predicted_pCR_0.5": row["predicted"],
                    "correct_0.5": row["correct"],
                }
            )
    handles = [
        Patch(facecolor=OUTCOME_COLORS[0], label="Non-pCR"),
        Patch(facecolor=OUTCOME_COLORS[1], label="pCR"),
        plt.Line2D([], [], marker="x", color="#B2182B", lw=0, label="Misclassified at 0.5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "EV02", "Patient risk waterfalls", pd.DataFrame(source_rows))


def figure_clinical_fusion_scatter(oof: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    source = oof[["subject_id", "outer_fold", "pCR", "clinical_11", "trimodal_equal_weight_11"]].copy()
    source["fusion_minus_clinical"] = source["trimodal_equal_weight_11"] - source["clinical_11"]
    source["reclassified_at_0.5"] = (
        (source["clinical_11"] >= 0.5).astype(int) != (source["trimodal_equal_weight_11"] >= 0.5).astype(int)
    )
    for label, value in [("Non-pCR", 0), ("pCR", 1)]:
        frame = source[source["pCR"] == value]
        ax.scatter(
            frame["clinical_11"],
            frame["trimodal_equal_weight_11"],
            s=30,
            alpha=0.75,
            color=OUTCOME_COLORS[value],
            edgecolor="white",
            linewidth=0.45,
            label=f"{label} (n={len(frame)})",
        )
    reclassified = source[source["reclassified_at_0.5"]]
    ax.scatter(
        reclassified["clinical_11"],
        reclassified["trimodal_equal_weight_11"],
        s=80,
        facecolors="none",
        edgecolors="#B2182B",
        linewidths=1.1,
        label="0.5-threshold reclassification",
    )
    ax.plot([0, 1], [0, 1], color="#000000", lw=1, ls=":")
    ax.axvline(0.5, color="#000000", lw=0.8, ls="--")
    ax.axhline(0.5, color="#000000", lw=0.8, ls="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Clinical-model predicted probability", ylabel="Trimodal predicted probability")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color="#ECECEC", lw=0.6)
    fig.tight_layout()
    save_figure(fig, "EV03", "Clinical versus trimodal probabilities", source)


def figure_fold_auc(per_fold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.7), sharex=True, sharey=True)
    axes = axes.ravel()
    source_rows = []
    for ax, (key, label, short) in zip(axes, MODEL_ORDER):
        frame = per_fold[per_fold["model_key"] == key].sort_values("fold")
        f = frame["fold"].to_numpy(); a = frame["auc"].to_numpy()
        ax.plot(f, a, color=MODEL_COLORS[key], lw=1.7, zorder=2)
        ax.scatter(f, a, s=44, color=MODEL_COLORS[key], edgecolors="white", linewidths=0.7, zorder=3)
        mean = float(a.mean()); sd = float(a.std(ddof=1))
        ax.axhline(mean, color="#000000", lw=1.1, ls="--", zorder=1)
        ax.set_title(label + f"\nmean {mean:.3f}; SD {sd:.3f}", fontsize=9.2, fontweight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.45, 0.95)
        for row in frame[["fold", "auc"]].to_dict("records"):
            source_rows.append({"model_key": key, "model_label": label, "fold": row["fold"], "auc": row["auc"]})
    for ax in axes[3:]:
        ax.set_xlabel("Outer fold")
    for ax in axes[::3]:
        ax.set_ylabel("AUC")
    fig.tight_layout()
    save_figure(fig, "EV04", "Five fold AUC stability", pd.DataFrame(source_rows))


def figure_probability_distributions(oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 6.6), sharex=True, sharey=True)
    source_rows = []
    rng = np.random.default_rng(7)
    for panel, (ax, (key, label, short)) in zip("ABCDEF", zip(axes.flat, MODEL_ORDER)):
        data = [oof.loc[oof["pCR"] == outcome, key].to_numpy() for outcome in (0, 1)]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showextrema=False)
        for body, outcome in zip(parts["bodies"], (0, 1)):
            body.set_facecolor(OUTCOME_COLORS[outcome])
            body.set_edgecolor("white")
            body.set_alpha(0.55)
        ax.boxplot(
            data,
            positions=[0, 1],
            widths=0.16,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#000000", "lw": 1},
            boxprops={"facecolor": "white", "edgecolor": "#444444", "lw": 0.7},
            whiskerprops={"color": "#444444", "lw": 0.7},
            capprops={"color": "#444444", "lw": 0.7},
        )
        for position, outcome, values in zip([0, 1], [0, 1], data):
            ax.scatter(
                position + rng.uniform(-0.075, 0.075, len(values)),
                values,
                s=8,
                color=OUTCOME_COLORS[outcome],
                alpha=0.65,
                linewidth=0,
            )
            for value in values:
                source_rows.append(
                    {
                        "model_key": key,
                        "model_label": label,
                        "outcome": "pCR" if outcome else "non-pCR",
                        "probability": value,
                    }
                )
        ax.axhline(0.5, color="#000000", lw=0.8, ls="--")
        ax.set_title(short)
        ax.set_xticks([0, 1], ["Non-pCR", "pCR"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#ECECEC", lw=0.55)
        add_panel_label(ax, panel)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted probability")
    fig.tight_layout()
    save_figure(fig, "EV05", "Probability distributions by outcome", pd.DataFrame(source_rows))


def figure_confusion_matrices(oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 6.3))
    source_rows = []
    for panel, (ax, (key, label, short)) in zip("ABCDEF", zip(axes.flat, MODEL_ORDER)):
        matrix = confusion_matrix(oof["pCR"], (oof[key] >= 0.5).astype(int), labels=[0, 1])
        row_percent = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n({row_percent[row, column] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if row_percent[row, column] > 0.52 else "#000000",
                    fontsize=9,
                )
        ax.set_title(short)
        ax.set_xticks([0, 1], ["Pred. non-pCR", "Pred. pCR"])
        ax.set_yticks([0, 1], ["Obs. non-pCR", "Obs. pCR"])
        ax.tick_params(axis="y", which="major", pad=5, labelrotation=90)
        for tick_label in ax.get_yticklabels():
            tick_label.set_rotation(90)
            tick_label.set_horizontalalignment("center")
            tick_label.set_verticalalignment("center")
        ax.set_xlabel("")
        ax.set_ylabel("")
        add_panel_label(ax, panel)
        for observed in (0, 1):
            for predicted in (0, 1):
                source_rows.append(
                    {
                        "model_key": key,
                        "model_label": label,
                        "threshold": 0.5,
                        "observed": observed,
                        "predicted": predicted,
                        "count": int(matrix[observed, predicted]),
                        "row_percent": float(row_percent[observed, predicted]),
                    }
                )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.95, wspace=0.34, hspace=0.12)
    cbar_ax = fig.add_axes([0.91, 0.22, 0.018, 0.56])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label("Row-normalized proportion")
    save_figure(fig, "EV06", "Confusion matrices at threshold 0.5", pd.DataFrame(source_rows))


def figure_threshold_curves(oof: pd.DataFrame) -> None:
    thresholds = np.linspace(0.05, 0.9, 171)
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 6.6), sharex=True, sharey=True)
    source_rows = []
    y = oof["pCR"].to_numpy()
    for panel, (ax, (key, label, short)) in zip("ABCDEF", zip(axes.flat, MODEL_ORDER)):
        p = oof[key].to_numpy()
        sensitivity = np.array([np.mean(p[y == 1] >= t) for t in thresholds])
        specificity = np.array([np.mean(p[y == 0] < t) for t in thresholds])
        youden = sensitivity + specificity - 1
        best = int(np.argmax(youden))
        ax.plot(thresholds, sensitivity, color="#B2182B", lw=1.6, label="Sensitivity")
        ax.plot(thresholds, specificity, color="#2166AC", lw=1.6, label="Specificity")
        ax.scatter(thresholds[best], sensitivity[best], s=24, color="#B2182B", zorder=3)
        ax.scatter(thresholds[best], specificity[best], s=24, color="#2166AC", zorder=3)
        ax.axvline(0.5, color="#444444", lw=0.8, ls="--")
        ax.set_title(f"{short}  Youden threshold={thresholds[best]:.2f}")
        ax.set_ylim(0, 1.02)
        ax.grid(color="#ECECEC", lw=0.55)
        add_panel_label(ax, panel)
        for threshold, sens, spec in zip(thresholds, sensitivity, specificity):
            source_rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "threshold": threshold,
                    "sensitivity": sens,
                    "specificity": spec,
                    "youden_index": sens + spec - 1,
                }
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    for ax in axes[:, 0]:
        ax.set_ylabel("Metric value")
    for ax in axes[-1, :]:
        ax.set_xlabel("Probability threshold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(
        fig,
        "EV07",
        "Threshold sensitivity and specificity",
        pd.DataFrame(source_rows),
        notes="Optimal thresholds are descriptive and not prespecified clinical cutoffs.",
    )


def figure_paired_bootstrap(oof: pd.DataFrame) -> None:
    y = oof["pCR"].to_numpy()
    folds = oof["outer_fold"].to_numpy()
    comparisons = [
        ("Clinical vs MRI", "clinical_11", "mri", True),
        ("Clinical vs pathology", "clinical_11", "pathology", True),
        ("Trimodal fusion vs clinical", "trimodal_equal_weight_11", "clinical_11", True),
        ("Clinical+MRI vs clinical", "clinical_plus_mri", "clinical_11", False),
        ("Clinical+pathology vs clinical", "clinical_plus_pathology", "clinical_11", False),
        ("Trimodal vs clinical+MRI", "trimodal_equal_weight_11", "clinical_plus_mri", False),
        ("Trimodal vs clinical+pathology", "trimodal_equal_weight_11", "clinical_plus_pathology", False),
        ("MRI vs pathology", "mri", "pathology", False),
    ]
    existing = pd.read_csv(ROOT / "pairwise_auc_comparisons.csv")
    existing_rows = {row["Comparison"]: row for row in existing.to_dict("records")}
    rows = []
    for index, (label, key_a, key_b, use_existing) in enumerate(comparisons):
        if use_existing and label in existing_rows:
            row = existing_rows[label]
            ci_text = row["95% bootstrap CI"]
            lower, upper = [float(value.strip()) for value in ci_text.split("to")]
            estimate = float(row["AUC difference"])
            p_value = float(row["P value"])
            provenance = "Preserved from archived Table 3 bootstrap"
        else:
            estimate, lower, upper, p_value = paired_bootstrap_difference(
                y,
                oof[key_a].to_numpy(),
                oof[key_b].to_numpy(),
                folds,
                n_boot=5000,
                seed=20261103 + index,
            )
            provenance = "New paired stratified bootstrap, exploratory"
        rows.append(
            {
                "comparison": label,
                "model_a": key_a,
                "model_b": key_b,
                "auc_difference": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "provenance": provenance,
            }
        )
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    y_positions = np.arange(len(source))[::-1]
    for position, row in zip(y_positions, source.to_dict("records")):
        color = "#B2182B" if row["ci_lower"] <= 0 <= row["ci_upper"] else "#2166AC"
        ax.plot([row["ci_lower"], row["ci_upper"]], [position, position], color=color, lw=2)
        ax.scatter(row["auc_difference"], position, color=color, s=34, zorder=3)
        ax.text(0.345, position, f"p={row['p_value']:.3f}", va="center", fontsize=7.5, color="#444444")
    ax.axvline(0, color="#000000", lw=0.9, ls="--")
    ax.set_yticks(y_positions, source["comparison"])
    ax.set(xlim=(-0.12, 0.38), xlabel="AUC difference (95% paired bootstrap CI)")
    ax.grid(axis="x", color="#E8E8E8", lw=0.55)
    fig.tight_layout()
    save_figure(
        fig,
        "EV08",
        "Paired bootstrap AUC differences",
        source,
        evidence="Table 3 values plus exploratory paired bootstrap comparisons",
        notes="The first three rows preserve archived Table 3 values; additional rows are exploratory.",
    )


def figure_dca_delta(oof: pd.DataFrame) -> None:
    thresholds = np.linspace(0.05, 0.4, 141)
    y = oof["pCR"].to_numpy()
    folds = oof["outer_fold"].to_numpy()
    p_fusion = oof["trimodal_equal_weight_11"].to_numpy()
    p_clinical = oof["clinical_11"].to_numpy()
    observed = net_benefit(y, p_fusion, thresholds) - net_benefit(y, p_clinical, thresholds)
    rng = np.random.default_rng(20261103)
    fold_indices = [np.flatnonzero(folds == fold) for fold in np.unique(folds)]
    draws = np.empty((1000, len(thresholds)), dtype=float)
    for index in range(len(draws)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        draws[index] = net_benefit(y[sampled], p_fusion[sampled], thresholds) - net_benefit(
            y[sampled], p_clinical[sampled], thresholds
        )
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    ax.fill_between(thresholds, lower, upper, color="#B7D7E8", alpha=0.6, linewidth=0, label="95% bootstrap CI")
    ax.plot(thresholds, observed, color="#1F5A7A", lw=1.8, label="Trimodal minus clinical")
    ax.axhline(0, color="#000000", lw=0.9, ls="--")
    ax.set(xlabel="Threshold probability", ylabel="Delta net benefit", xlim=(0.05, 0.4))
    ax.grid(color="#ECECEC", lw=0.55)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    source = pd.DataFrame(
        {
            "threshold": thresholds,
            "delta_net_benefit": observed,
            "ci_lower": lower,
            "ci_upper": upper,
            "bootstrap_resamples": 1000,
        }
    )
    save_figure(
        fig,
        "EV09",
        "DCA delta net benefit",
        source,
        evidence="Exploratory paired bootstrap of frozen OOF predictions",
    )


def figure_model_correlation(oof: pd.DataFrame) -> None:
    keys = [key for key, _, _ in MODEL_ORDER]
    labels = [short for _, _, short in MODEL_ORDER]
    matrix = oof[keys].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    for row in range(len(keys)):
        for column in range(len(keys)):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.6 else "#000000",
                fontsize=8,
            )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Spearman correlation")
    fig.tight_layout()
    source = matrix.reset_index(names="model")
    source.columns = ["model"] + labels
    save_figure(fig, "EV10", "Model prediction correlation", source)


def figure_continuous_characteristics(dataset: pd.DataFrame) -> None:
    variables = [
        ("Age_years", "Age (years)"),
        ("ER_percent", "ER (%)"),
        ("PR_percent", "PR (%)"),
        ("Ki67_percent", "Ki67 (%)"),
        ("tumour_volume_V1_mL", "Tumour volume V1 (mL)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 4.2))
    rng = np.random.default_rng(11)
    source_rows = []
    for panel, (ax, (column, label)) in zip("ABCDE", zip(axes, variables)):
        data = [dataset.loc[dataset["pCR"] == outcome, column].dropna().to_numpy() for outcome in (0, 1)]
        frames = {outcome: dataset.loc[dataset["pCR"] == outcome, ["subject_id", "pCR", column]].dropna() for outcome in (0, 1)}
        parts = ax.violinplot(data, positions=[0, 1], widths=0.68, showextrema=False)
        for body, outcome in zip(parts["bodies"], (0, 1)):
            body.set_facecolor(OUTCOME_COLORS[outcome])
            body.set_edgecolor("white")
            body.set_alpha(0.45)
        ax.boxplot(
            data,
            positions=[0, 1],
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#000000", "lw": 1},
            boxprops={"facecolor": "white", "edgecolor": "#444444", "lw": 0.65},
            whiskerprops={"color": "#444444", "lw": 0.65},
            capprops={"color": "#444444", "lw": 0.65},
        )
        for position, outcome, values in zip([0, 1], [0, 1], data):
            ax.scatter(
                position + rng.uniform(-0.07, 0.07, len(values)),
                values,
                s=8,
                color=OUTCOME_COLORS[outcome],
                alpha=0.7,
                linewidth=0,
            )
            for subject_id, value in zip(frames[outcome]["subject_id"], values):
                source_rows.append(
                    {
                        "variable": column,
                        "variable_label": label,
                        "subject_id": subject_id,
                        "pCR": outcome,
                        "value": value,
                    }
                )
        ax.set_xticks([0, 1], ["Non-pCR", "pCR"])
        ax.set_title(label)
        ax.grid(axis="y", color="#ECECEC", lw=0.55)
        add_panel_label(ax, panel)
    fig.tight_layout()
    save_figure(
        fig,
        "EV11",
        "Continuous characteristics by pCR",
        pd.DataFrame(source_rows),
        evidence="Descriptive patient-level baseline data",
    )


def figure_categorical_composition(dataset: pd.DataFrame) -> None:
    dataset = dataset.copy()
    dataset["ER_status"] = np.where(dataset["ER_percent"] >= 1, "Positive", "Negative")
    dataset["anti_HER2"] = dataset["anti_HER2_explicit_H"].map({1: "Yes", 0: "No"})
    specs = [
        ("HER2_IHC", "HER2 IHC", ["0", "1", "2", "3"]),
        ("ER_status", "ER status", ["Negative", "Positive"]),
        ("clinical_N_stage", "N stage", ["0", "1", "2", "3"]),
        ("anti_HER2", "Anti-HER2 regimen", ["No", "Yes"]),
    ]
    palettes = ["#4477AA", "#66CCEE", "#228833", "#CCBB44"]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 4.2))
    source_rows = []
    for panel, (ax, (column, title, order)) in zip("ABCD", zip(axes, specs)):
        if column == "HER2_IHC":
            dataset[column] = dataset[column].astype(int).astype(str)
        group_sizes = dataset["pCR"].value_counts().reindex([0, 1]).astype(int)
        proportions = []
        for outcome in (0, 1):
            frame = dataset[dataset["pCR"] == outcome]
            counts = frame[column].astype(str).value_counts().reindex(order, fill_value=0)
            proportions.append((counts / len(frame)).to_numpy())
        proportions = np.asarray(proportions)
        bottoms = np.zeros(2)
        for level_index, level in enumerate(order):
            values = proportions[:, level_index] * 100
            ax.bar([0, 1], values, bottom=bottoms, color=palettes[level_index], width=0.7, label=level)
            for position in (0, 1):
                if values[position] >= 6:
                    ax.text(position, bottoms[position] + values[position] / 2, f"{values[position]:.1f}%", ha="center", va="center", fontsize=7, color="white")
                source_rows.append(
                    {
                        "variable": column,
                        "variable_label": title,
                        "outcome": "pCR" if position else "non-pCR",
                        "level": level,
                        "proportion": proportions[position, level_index],
                    }
                )
            bottoms += values
        ax.set_xticks([0, 1], [f"Non-pCR\nn={group_sizes[0]}", f"pCR\nn={group_sizes[1]}"])
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel("Composition (%)" if panel == "A" else "")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.8)
        add_panel_label(ax, panel)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(
        fig,
        "EV12",
        "Categorical composition by pCR",
        pd.DataFrame(source_rows),
        evidence="Descriptive patient-level baseline data",
    )


def figure_her2_rates(dataset: pd.DataFrame) -> None:
    rows = []
    for ihc in sorted(dataset["HER2_IHC"].dropna().astype(int).unique()):
        frame = dataset[dataset["HER2_IHC"] == ihc]
        success = int(frame["pCR"].sum())
        total = len(frame)
        lower, upper = wilson_interval(success, total)
        rows.append(
            {
                "HER2_IHC": f"{ihc}+",
                "pCR": success,
                "total": total,
                "pCR_rate": success / total,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
            }
        )
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4.1))
    positions = np.arange(len(source))
    ax.bar(positions, source["pCR_rate"] * 100, color="#4C9FCE", width=0.65)
    ax.errorbar(
        positions,
        source["pCR_rate"] * 100,
        yerr=[
            (source["pCR_rate"] - source["wilson_95_lower"]) * 100,
            (source["wilson_95_upper"] - source["pCR_rate"]) * 100,
        ],
        fmt="none",
        ecolor="#000000",
        capsize=3,
        lw=1,
    )
    for position, row in zip(positions, source.to_dict("records")):
        label_y = row["wilson_95_upper"] * 100 + 2.5
        ax.text(position, label_y, f"{row['pCR']}/{row['total']}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, source["HER2_IHC"])
    y_max = max(70.0, float(source["wilson_95_upper"].max() * 100) + 9.0)
    ax.set(xlabel="HER2 immunohistochemistry", ylabel="pCR rate (%)", ylim=(0, y_max))
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    fig.tight_layout()
    save_figure(
        fig,
        "EV13",
        "HER2 IHC stratified pCR rate",
        source,
        evidence="Descriptive proportions with Wilson 95% confidence intervals",
        notes="IHC 2+ is not equated with HER2 positivity because ISH results are unavailable.",
    )


def figure_clinical_correlation(dataset: pd.DataFrame) -> None:
    columns = [
        "Age_years",
        "ER_percent",
        "PR_percent",
        "Ki67_percent",
        "HER2_IHC",
        "histological_grade",
        "clinical_T_stage",
        "clinical_N_stage",
        "V1_log",
        "anti_HER2_explicit_H",
        "pCR",
    ]
    labels = ["Age", "ER", "PR", "Ki67", "HER2", "Grade", "T", "N", "log(V1)", "Anti-HER2", "pCR"]
    matrix = dataset[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    for row in range(len(columns)):
        for column in range(len(columns)):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.62 else "#000000",
                fontsize=6.8,
            )
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Spearman rho")
    fig.tight_layout()
    source = matrix.reset_index(names="variable")
    source.columns = ["variable"] + labels
    save_figure(
        fig,
        "EV14",
        "Clinical variable correlation",
        source,
        evidence="Exploratory Spearman correlations",
        notes="Pairwise correlations are not causal and are unstable in small subgroups.",
    )


def figure_subgroup_forest(dataset: pd.DataFrame, oof: pd.DataFrame) -> None:
    merged = oof.merge(dataset, on=["subject_id", "pCR"], how="inner")
    merged["ER_status"] = np.where(merged["ER_percent"] >= 1, "ER+", "ER-")
    merged["N_status"] = np.where(merged["clinical_N_stage"] == 0, "N0", "N1+")
    merged["Menopause"] = merged["menopausal_status_0_1"].map({0: "Pre-menopausal", 1: "Post-menopausal"})
    merged["Anti-HER2"] = merged["anti_HER2_explicit_H"].map({0: "No", 1: "Yes"})
    merged["HER2"] = merged["HER2_IHC"].astype(int).map(lambda value: f"IHC {value}+")
    subgroups = [
        ("Overall", np.repeat("All patients", len(merged))),
        ("HER2 IHC", merged["HER2"].to_numpy()),
        ("ER status", merged["ER_status"].to_numpy()),
        ("N stage", merged["N_status"].to_numpy()),
        ("Menopause", merged["Menopause"].to_numpy()),
        ("Anti-HER2", merged["Anti-HER2"].to_numpy()),
    ]
    models = [("clinical_11", "Clinical (11)"), ("trimodal_equal_weight_11", "Trimodal")]
    rng = np.random.default_rng(20261103)
    rows = []
    for model_key, model_label in models:
        for variable, values in subgroups:
            for value in sorted(np.unique(values)):
                mask = values == value
                y = merged.loc[mask, "pCR"].to_numpy()
                p = merged.loc[mask, model_key].to_numpy()
                if len(y) < 8 or y.sum() < 3 or (1 - y).sum() < 3:
                    continue
                draws = []
                for _ in range(1000):
                    indices = rng.choice(np.arange(len(y)), size=len(y), replace=True)
                    if len(np.unique(y[indices])) == 2:
                        draws.append(roc_auc_score(y[indices], p[indices]))
                estimate = roc_auc_score(y, p)
                lower, upper = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": model_label,
                        "subgroup_variable": variable,
                        "subgroup": value,
                        "n": len(y),
                        "events_pCR": int(y.sum()),
                        "auc": estimate,
                        "bootstrap_95_lower": lower,
                        "bootstrap_95_upper": upper,
                        "bootstrap_resamples": len(draws),
                    }
                )
    source = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 7.0), sharey=True)
    labels = (source["subgroup_variable"] + ": " + source["subgroup"]).drop_duplicates().tolist()
    counts = source.drop_duplicates("subgroup").set_index("subgroup")[["n", "events_pCR"]]
    display_labels = [
        f"{label}\nn={int(counts.loc[label.split(': ', 1)[1], 'n'])}; pCR={int(counts.loc[label.split(': ', 1)[1], 'events_pCR'])}"
        for label in labels
    ]
    y_positions = {label: index for index, label in enumerate(labels[::-1])}
    for panel, (ax, (model_key, model_label)) in zip("AB", zip(axes, models)):
        frame = source[source["model_key"] == model_key]
        for row in frame.to_dict("records"):
            label = f"{row['subgroup_variable']}: {row['subgroup']}"
            position = y_positions[label]
            ax.plot([row["bootstrap_95_lower"], row["bootstrap_95_upper"]], [position, position], color=MODEL_COLORS[model_key], lw=1.6)
            ax.scatter(row["auc"], position, color=MODEL_COLORS[model_key], s=26, zorder=3)
        ax.axvline(0.5, color="#000000", lw=0.8, ls="--")
        ax.set(xlim=(0.2, 1.02), xlabel="AUC (95% bootstrap CI)")
        ax.set_title(model_label)
        ax.grid(axis="x", color="#ECECEC", lw=0.5)
        add_panel_label(ax, panel)
    axes[0].set_yticks(range(len(display_labels)), display_labels[::-1])
    axes[0].tick_params(axis="y", labelsize=6.8)
    fig.tight_layout()
    save_figure(
        fig,
        "EV15",
        "Subgroup AUC forest plot",
        source,
        evidence="Exploratory subgroup analysis",
        notes="Only groups with at least 8 patients, 3 pCR and 3 non-pCR events are shown.",
    )


def figure_pca(dataset: pd.DataFrame) -> None:
    columns = [
        "Age_years",
        "ER_percent",
        "PR_percent",
        "Ki67_percent",
        "HER2_IHC",
        "menopausal_status_0_1",
        "histological_grade",
        "clinical_T_stage",
        "clinical_N_stage",
        "V1_log",
        "anti_HER2_explicit_H",
    ]
    labels = ["Age", "ER", "PR", "Ki67", "HER2", "Meno", "Grade", "T", "N", "log(V1)", "Anti-HER2"]
    x = StandardScaler().fit_transform(dataset[columns].astype(float))
    pca = PCA(n_components=2)
    scores = pca.fit_transform(x)
    source = dataset[["subject_id", "pCR"]].copy()
    source["PC1"] = scores[:, 0]
    source["PC2"] = scores[:, 1]
    for label, loading in zip(labels, pca.components_.T):
        source[f"loading_PC1_{label}"] = float(loading[0])
        source[f"loading_PC2_{label}"] = float(loading[1])
    fig, (ax, loading_ax) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for outcome, label in [(0, "Non-pCR"), (1, "pCR")]:
        frame = source[source["pCR"] == outcome]
        ax.scatter(frame["PC1"], frame["PC2"], s=30, color=OUTCOME_COLORS[outcome], alpha=0.75, edgecolor="white", linewidth=0.4, label=f"{label} (n={len(frame)})")
    ax.axhline(0, color="#AAAAAA", lw=0.6)
    ax.axvline(0, color="#AAAAAA", lw=0.6)
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")
    loadings = pca.components_[:2].T
    for label, (x_value, y_value) in zip(labels, loadings):
        if np.hypot(x_value, y_value) >= 0.32:
            loading_ax.arrow(0, 0, x_value, y_value, color="#000000", width=0.006, head_width=0.045, length_includes_head=True)
            label_positions = {
                "Age": (x_value - 0.06, y_value + 0.05),
                "Meno": (x_value + 0.05, y_value - 0.06),
            }
            text_x, text_y = label_positions.get(label, (x_value * 1.08, y_value * 1.08))
            loading_ax.text(text_x, text_y, label, fontsize=7)
    loading_ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), xlabel="PC1 loading", ylabel="PC2 loading")
    loading_ax.axhline(0, color="#AAAAAA", lw=0.6)
    loading_ax.axvline(0, color="#AAAAAA", lw=0.6)
    loading_ax.set_aspect("equal", adjustable="box")
    add_panel_label(loading_ax, "B")
    fig.tight_layout()
    save_figure(
        fig,
        "EV16",
        "Clinical feature PCA",
        source,
        evidence="Exploratory unsupervised transformation",
        notes="PCA axes are linear combinations and do not establish biologic clusters.",
    )


def figure_sample_size(oof: pd.DataFrame) -> None:
    y = oof["pCR"].to_numpy()
    p = oof["trimodal_equal_weight_11"].to_numpy()
    sample_sizes = np.arange(40, 106, 5)
    rng = np.random.default_rng(20261103)
    rows = []
    for sample_size in sample_sizes:
        draws = []
        for _ in range(2000):
            indices = rng.choice(len(y), size=sample_size, replace=False)
            if len(np.unique(y[indices])) == 2:
                draws.append(roc_auc_score(y[indices], p[indices]))
        draws = np.asarray(draws)
        rows.append(
            {
                "sample_size": sample_size,
                "resamples": len(draws),
                "auc_mean": draws.mean(),
                "auc_median": np.median(draws),
                "auc_2.5": np.quantile(draws, 0.025),
                "auc_97.5": np.quantile(draws, 0.975),
            }
        )
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(source["sample_size"], source["auc_2.5"], source["auc_97.5"], color="#B7D7E8", alpha=0.65, linewidth=0, label="95% resampling interval")
    ax.plot(source["sample_size"], source["auc_median"], color="#1F5A7A", lw=1.8, marker="o", ms=3.5, label="Median AUC")
    full_auc = roc_auc_score(y, p)
    ax.axhline(full_auc, color="#B2182B", lw=1, ls="--", label=f"Full OOF AUC={full_auc:.3f}")
    ax.set(xlabel="Resampled cohort size", ylabel="Trimodal AUC", xlim=(38, 106))
    ax.grid(color="#ECECEC", lw=0.55)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(
        fig,
        "EV17",
        "AUC precision by cohort size",
        source,
        evidence="Resampling simulation from frozen OOF predictions",
        notes="This is a precision simulation, not a new training learning curve.",
    )


def figure_permutation_importance(dataset: pd.DataFrame, oof: pd.DataFrame) -> None:
    model_dir_value = os.environ.get("NAC_MODEL_DIR", "").strip()
    if not model_dir_value:
        if not PERMUTATION_SOURCE.exists():
            raise FileNotFoundError(
                "EV18 source data are missing. Set NAC_MODEL_DIR to a local model "
                "directory containing checkpoints/ or restore the archived EV18 CSV."
            )
        render_permutation_importance(pd.read_csv(PERMUTATION_SOURCE))
        return
    import torch
    import torch.nn as nn
    import sys

    model_dir = Path(model_dir_value).expanduser().resolve()
    sys.path.insert(0, str(model_dir))
    import config as cfg
    from models import ClinicalMLP

    features = ["Age", "ER", "PR", "Ki67", "HER2", "Meno", "Grade", "T", "N", "V1_log", "NAC_HER2"]
    x = np.column_stack(
        [
            dataset["Age_years"] / 100,
            dataset["ER_percent"] / 100,
            dataset["PR_percent"] / 100,
            dataset["Ki67_percent"] / 100,
            dataset["HER2_IHC"] / 3,
            dataset["menopausal_status_0_1"],
            (dataset["histological_grade"] - 1) / 2,
            (dataset["clinical_T_stage"] - 1) / 3,
            dataset["clinical_N_stage"] / 3,
            dataset["V1_log"],
            dataset["anti_HER2_explicit_H"],
        ]
    ).astype(np.float32)
    y = dataset["pCR"].to_numpy()
    folds = oof.set_index("subject_id").loc[dataset["subject_id"], "outer_fold"].to_numpy()
    baseline_auc = roc_auc_score(y, oof["clinical_11"].to_numpy())
    rng = np.random.default_rng(20261103)
    n_repeats = 100
    rows = []
    fold_models = {}
    for fold in range(1, 6):
        state = torch.load(model_dir / "checkpoints" / f"fold{fold - 1}_clinical.pt", map_location="cpu")
        model = nn.Sequential(ClinicalMLP(input_dim=11).mlp, nn.Linear(16, 2))
        model.load_state_dict(state)
        model.eval()
        fold_models[fold] = model
    for feature_index, feature in enumerate(features):
        permuted_aucs = []
        for repeat in range(n_repeats):
            probabilities = np.empty(len(dataset), dtype=float)
            for fold in range(1, 6):
                val = np.flatnonzero(folds == fold)
                x_fold = x[val].copy()
                x_fold[:, feature_index] = rng.permutation(x_fold[:, feature_index])
                with torch.no_grad():
                    logits = fold_models[fold](torch.from_numpy(x_fold))
                    probabilities[val] = torch.softmax(logits, dim=1)[:, 1].numpy()
            permuted_aucs.append(roc_auc_score(y, probabilities))
        permuted_aucs = np.asarray(permuted_aucs)
        importance = baseline_auc - permuted_aucs
        rows.append(
            {
                "feature": feature,
                "baseline_oof_auc": baseline_auc,
                "permuted_auc_mean": permuted_aucs.mean(),
                "permuted_auc_sd": permuted_aucs.std(ddof=1),
                "auc_drop_mean": importance.mean(),
                "auc_drop_2.5": np.quantile(importance, 0.025),
                "auc_drop_97.5": np.quantile(importance, 0.975),
                "n_permutations": n_repeats,
            }
        )
    source = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    positions = np.arange(len(source))
    ax.errorbar(
        source["auc_drop_mean"],
        positions,
        xerr=[
            source["auc_drop_mean"] - source["auc_drop_2.5"],
            source["auc_drop_97.5"] - source["auc_drop_mean"],
        ],
        fmt="o",
        color="#006BA6",
        ecolor="#000000",
        capsize=2.5,
        ms=4,
        lw=1,
    )
    ax.axvline(0, color="#000000", lw=0.8, ls="--")
    ax.set_yticks(positions, source["feature"])
    ax.set(xlabel="Decrease in pooled OOF AUC after permutation")
    ax.grid(axis="x", color="#ECECEC", lw=0.55)
    fig.tight_layout()
    save_figure(
        fig,
        "EV18",
        "Clinical permutation importance",
        source,
        evidence="Frozen-model permutation analysis on validation folds",
        notes="Permutation importance reflects the fitted model and can split credit among correlated variables.",
    )


def render_permutation_importance(source: pd.DataFrame) -> None:
    source = source.sort_values("auc_drop_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    positions = np.arange(len(source))
    ax.errorbar(
        source["auc_drop_mean"],
        positions,
        xerr=[
            source["auc_drop_mean"] - source["auc_drop_2.5"],
            source["auc_drop_97.5"] - source["auc_drop_mean"],
        ],
        fmt="o",
        color="#006BA6",
        ecolor="#000000",
        capsize=2.5,
        ms=4,
        lw=1,
    )
    ax.axvline(0, color="#000000", lw=0.8, ls="--")
    ax.set_yticks(positions, source["feature"])
    ax.set(xlabel="Decrease in pooled OOF AUC after permutation")
    ax.grid(axis="x", color="#ECECEC", lw=0.55)
    fig.tight_layout()
    save_figure(
        fig,
        "EV18",
        "Clinical permutation importance",
        source,
        evidence="Frozen-model permutation analysis on validation folds",
        notes="Permutation importance reflects the fitted model and can split credit among correlated variables.",
    )


def write_contact_sheet() -> None:
    items = [row for row in INDEX_ROWS if str(row["png"]).endswith(".png")]
    if not items:
        return
    columns = 3
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 4.4 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items):
        image = plt.imread(OUT / str(item["png"]))
        ax.imshow(image)
        ax.set_title(f"{item['figure_id']} | {item['title']}", fontsize=9, loc="left")
        ax.axis("off")
    figure.suptitle("Extended visualization contact sheet (EV01-EV18)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(OUT / "VISUALIZATION_CONTACT_SHEET.png", dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT / "VISUALIZATION_CONTACT_SHEET.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_reference_tables(dataset: pd.DataFrame) -> None:
    dataset[["subject_id", "pCR"]].to_csv(TABLE_DIR / "visualization_patient_order.csv", index=False, encoding="utf-8-sig")
    model_table_columns = pd.DataFrame(
        [{"model_key": key, "model_label": label, "short_label": short, "color": MODEL_COLORS[key]} for key, label, short in MODEL_ORDER]
    )
    model_table_columns.to_csv(TABLE_DIR / "visualization_model_legend.csv", index=False, encoding="utf-8-sig")


def write_outputs() -> None:
    index = pd.DataFrame(INDEX_ROWS)
    index.to_csv(OUT / "VISUALIZATION_INDEX.csv", index=False, encoding="utf-8-sig")
    readme = """# Extended Visualization Suite

This directory contains additional candidate figures derived from the frozen,
deidentified 105-patient out-of-fold predictions and baseline data.

- `figures/`: PNG, PDF and TIFF versions of the 18 completed figures `EV01-EV18`.
- `figure_source/`: one CSV source-data file for every completed figure.
- `tables/`: model legend and patient-order reference tables.
- `VISUALIZATION_ATLAS.pdf`: all 18 completed figures in one review PDF.
- `VISUALIZATION_INDEX.csv`: figure inventory, evidence level and source-data link.
- `VISUALIZATION_CONTACT_SHEET.png/.pdf`: an overview of all 18 figures.

`EV18` uses the archived permutation source CSV when model checkpoints are not
available. Local checkpoints are not included in this package. Users with
authorized access may set `NAC_MODEL_DIR` to a local model directory that
contains `checkpoints/` to recompute EV18 from the fitted clinical models.

Interpretation guardrails:

1. These are exploratory secondary visualizations. The manuscript's primary
   tables and Figures 3-5/S1 remain the prespecified reporting basis.
2. The cohort contains 105 patients and 27 pCR events. Subgroup intervals are
   wide and should not be used to claim differential clinical utility.
3. DCA and threshold plots are descriptive. No threshold is proposed as a
   clinical decision rule.
4. Calibration is imperfect; high AUC does not imply that probabilities are
   ready for bedside use.
5. Imaging and pathology interpretation panels require separate privacy review
   and are not written into this public reproducibility directory.
6. Restricted pathology attention and MRI Grad-CAM figures are maintained in a
   separate local review directory. The MRI Grad-CAM panel is technical QC only
   and is not intended for manuscript submission.
7. The recommended manuscript additions are EV01 (precision-recall curves) and
   EV04 (five-fold AUC stability), followed by EV08 and EV02 if space permits.
8. EV09 does not show a stable fusion decision-curve advantage and must not be
   presented as evidence of clinical utility.
"""
    (OUT / "VISUALIZATION_README.md").write_text(readme, encoding="utf-8")
    ATLAS.close()


def main() -> None:
    dataset, oof, per_fold, _ = load_data()
    write_reference_tables(dataset)
    figure_pr_curves(oof)
    figure_risk_waterfall(oof)
    figure_clinical_fusion_scatter(oof)
    figure_fold_auc(per_fold)
    figure_probability_distributions(oof)
    figure_confusion_matrices(oof)
    figure_threshold_curves(oof)
    figure_paired_bootstrap(oof)
    figure_dca_delta(oof)
    figure_model_correlation(oof)
    figure_continuous_characteristics(dataset)
    figure_categorical_composition(dataset)
    figure_her2_rates(dataset)
    figure_clinical_correlation(dataset)
    figure_subgroup_forest(dataset, oof)
    figure_pca(dataset)
    figure_sample_size(oof)
    figure_permutation_importance(dataset, oof)
    write_contact_sheet()
    write_outputs()
    print(f"Created {len(INDEX_ROWS)} figures in {OUT}")


if __name__ == "__main__":
    main()
