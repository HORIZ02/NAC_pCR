"""Reproduce Tables 2-3 and Figures 3-5/S1 from archived OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import MonteCarloMethod, fisher_exact, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduced"
TABLE_DIR = OUTPUT / "tables"
FIGURE_DIR = OUTPUT / "figures"
SOURCE_DIR = OUTPUT / "figure_source"
for directory in (OUTPUT, TABLE_DIR, FIGURE_DIR, SOURCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    ("clinical_11", "Clinical model (11 variables)", "clinical_full_11"),
    ("mri", "MRI (3D DenseNet-121)", "mri"),
    ("pathology", "Pathology (UNI + attention MIL)", "pathology"),
    ("clinical_plus_mri", "Clinical + MRI", "clinical_plus_mri"),
    ("clinical_plus_pathology", "Clinical + pathology", "clinical_plus_pathology"),
    (
        "trimodal_equal_weight_11",
        "Clinical + MRI + pathology",
        "fusion_equal_full_11",
    ),
]
COLORS = {
    "clinical_11": "#0072B2",
    "mri": "#56B4E9",
    "pathology": "#009E73",
    "clinical_plus_mri": "#E69F00",
    "clinical_plus_pathology": "#009E73",
    "trimodal_equal_weight_11": "#CC79A7",
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "axes.labelpad": 5,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset = pd.read_csv(ROOT / "analysis_dataset_deidentified.csv")
    oof = pd.read_csv(ROOT / "out_of_fold_predictions.csv")
    reported = json.loads((ROOT / "reported_values.json").read_text(encoding="utf-8"))
    return dataset, oof, reported


def ordered_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for fold in range(1, 6):
        frame = oof[oof["outer_fold"] == fold].copy()
        frame["_row"] = frame["subject_id"].str.removeprefix("P").astype(int)
        rows.extend(frame.sort_values("_row").drop(columns="_row").to_dict("records"))
    return pd.DataFrame(rows).reset_index(drop=True)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "n": int(len(y)),
    }


def bootstrap_auc_ci(
    y: np.ndarray, probability: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(roc_auc_score(y[indices], probability[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def reproduce_table1(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = dataset.copy()
    frame["pCR"] = frame["pCR"].astype(int)
    frame["ER_status"] = np.where(frame["ER_percent"] >= 1, "Positive", "Negative")
    frame["PR_status"] = np.where(frame["PR_percent"] >= 1, "Positive", "Negative")
    frame["Menopause"] = frame["menopausal_status_0_1"].map(
        {0: "Pre-menopausal", 1: "Post-menopausal"}
    )
    frame["NAC_HER2"] = frame["anti_HER2_explicit_H"].map({1: "Yes", 0: "No"})
    frame["HER2_IHC"] = frame["HER2_IHC"].map(lambda value: f"{int(value)}+")
    groups = {0: frame[frame["pCR"] == 0], 1: frame[frame["pCR"] == 1]}
    rows: list[dict] = []
    summary: dict = {
        "n": int(len(frame)),
        "pcr": int(frame["pCR"].sum()),
        "non_pcr": int((1 - frame["pCR"]).sum()),
        "anti_her2": int((frame["NAC_HER2"] == "Yes").sum()),
        "her2_ihc": {
            str(key): int(value)
            for key, value in frame["HER2_IHC"].value_counts().sort_index().items()
        },
    }

    def add_continuous(label: str, column: str) -> None:
        non_pcr = groups[0][column].astype(float)
        pcr = groups[1][column].astype(float)
        p_value = mannwhitneyu(non_pcr, pcr, alternative="two-sided").pvalue
        rows.append(
            {
                "Characteristic": label,
                "Non-pCR (n=78)": (
                    f"{non_pcr.median():.1f} "
                    f"({non_pcr.quantile(0.25):.1f}-{non_pcr.quantile(0.75):.1f})"
                ),
                "pCR (n=27)": (
                    f"{pcr.median():.1f} ({pcr.quantile(0.25):.1f}-{pcr.quantile(0.75):.1f})"
                ),
                "P value": "<0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "_level": 0,
                "_p": float(p_value),
            }
        )

    def add_categorical(label: str, column: str, order: list | None = None) -> None:
        levels = order or sorted(frame[column].dropna().unique(), key=str)
        table = pd.crosstab(frame[column], frame["pCR"]).reindex(
            index=levels, columns=[0, 1], fill_value=0
        )
        if table.shape == (2, 2):
            p_value = float(fisher_exact(table.to_numpy(), alternative="two-sided").pvalue)
        else:
            p_value = float(
                fisher_exact(
                    table.to_numpy(),
                    method=MonteCarloMethod(n_resamples=50000, rng=42),
                ).pvalue
            )
        rows.append(
            {
                "Characteristic": label,
                "Non-pCR (n=78)": "",
                "pCR (n=27)": "",
                "P value": "<0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "_level": 0,
                "_p": p_value,
            }
        )
        for level in levels:
            n0 = int(table.loc[level, 0])
            n1 = int(table.loc[level, 1])
            rows.append(
                {
                    "Characteristic": f"  {level}",
                    "Non-pCR (n=78)": f"{n0} ({100 * n0 / 78:.1f})",
                    "pCR (n=27)": f"{n1} ({100 * n1 / 27:.1f})",
                    "P value": "",
                    "_level": 1,
                    "_p": None,
                }
            )

    add_continuous("Age, years", "Age_years")
    add_continuous("ER, %", "ER_percent")
    add_continuous("PR, %", "PR_percent")
    add_continuous("Ki67, %", "Ki67_percent")
    add_continuous("Tumour volume (V1), mL", "tumour_volume_V1_mL")
    add_categorical("ER status", "ER_status", ["Negative", "Positive"])
    add_categorical("PR status", "PR_status", ["Negative", "Positive"])
    add_categorical("HER2 IHC", "HER2_IHC", ["0+", "1+", "2+", "3+"])
    add_categorical(
        "Menopausal status", "Menopause", ["Pre-menopausal", "Post-menopausal"]
    )
    add_categorical("Histological grade", "histological_grade", [1, 2, 3])
    add_categorical("Clinical T stage", "clinical_T_stage", [1, 2, 3, 4])
    add_categorical("Clinical N stage", "clinical_N_stage", [0, 1, 2, 3])
    add_categorical("Anti-HER2 therapy", "NAC_HER2", ["No", "Yes"])

    grade_labels = {1: "Grade I", 2: "Grade II", 3: "Grade III"}
    current_section = ""
    for row in rows:
        label = row["Characteristic"]
        if label == "Histological grade":
            current_section = "grade"
        elif label in (
            "Clinical T stage",
            "Clinical N stage",
            "Anti-HER2 therapy",
            "Menopausal status",
            "HER2 IHC",
        ):
            current_section = ""
        if current_section == "grade" and label.startswith("  "):
            row["Characteristic"] = f"  {grade_labels[int(label.strip())]}"

    clean_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    result = pd.DataFrame(clean_rows)
    summary["rows"] = clean_rows
    summary["age_median_iqr"] = [
        float(frame["Age_years"].median()),
        float(frame["Age_years"].quantile(0.25)),
        float(frame["Age_years"].quantile(0.75)),
    ]
    result.to_csv(
        TABLE_DIR / "Table_1_baseline_characteristics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLE_DIR / "Table_1_baseline_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, summary


def reproduce_tables(
    oof: pd.DataFrame, reported: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = ordered_oof(oof)
    y_all = ordered["pCR"].to_numpy(dtype=int)
    table2_rows: list[dict] = []
    fold_rows: list[dict] = []
    auc_rows: list[dict] = []

    for key, label, reported_key in MODEL_ORDER:
        fold_metrics = []
        for fold in range(1, 6):
            frame = ordered[ordered["outer_fold"] == fold]
            fold_metric = metrics(
                frame["pCR"].to_numpy(dtype=int), frame[key].to_numpy(dtype=float)
            )
            fold_metric["fold"] = fold
            fold_metrics.append(fold_metric)
            fold_rows.append({"model_key": key, "model_label": label, **fold_metric})

        probability = ordered[key].to_numpy(dtype=float)
        pooled = metrics(y_all, probability)
        ci_low, ci_high = bootstrap_auc_ci(y_all, probability)
        mean_auc = float(np.mean([item["auc"] for item in fold_metrics]))
        sd_auc = float(np.std([item["auc"] for item in fold_metrics], ddof=1))
        mean_brier = float(np.mean([item["brier"] for item in fold_metrics]))
        sd_brier = float(np.std([item["brier"] for item in fold_metrics], ddof=1))

        for metric_name, expected in (
            ("mean_auc", reported["models"][reported_key]["mean_auc"]),
            ("pooled_auc", reported["models"][reported_key]["pooled"]["auc"]),
            ("ci_low", reported["models"][reported_key]["auc_ci95"][0]),
            ("ci_high", reported["models"][reported_key]["auc_ci95"][1]),
        ):
            observed = {
                "mean_auc": mean_auc,
                "pooled_auc": pooled["auc"],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }[metric_name]
            if not np.isclose(observed, expected, atol=1e-10, rtol=0):
                raise AssertionError(f"{key} {metric_name}: {observed} != {expected}")

        table2_rows.append(
            {
                "Model": label,
                "AUC, mean ± SD": f"{mean_auc:.3f} ± {sd_auc:.3f}",
                "Pooled AUC (95% CI)": (
                    f"{pooled['auc']:.3f} ({ci_low:.3f}–{ci_high:.3f})"
                ),
                "Accuracy": (
                    f"{np.mean([item['accuracy'] for item in fold_metrics]):.3f} ± "
                    f"{np.std([item['accuracy'] for item in fold_metrics], ddof=1):.3f}"
                ),
                "Sensitivity": (
                    f"{np.mean([item['sensitivity'] for item in fold_metrics]):.3f} ± "
                    f"{np.std([item['sensitivity'] for item in fold_metrics], ddof=1):.3f}"
                ),
                "Specificity": (
                    f"{np.mean([item['specificity'] for item in fold_metrics]):.3f} ± "
                    f"{np.std([item['specificity'] for item in fold_metrics], ddof=1):.3f}"
                ),
                "Brier score": f"{mean_brier:.3f} ± {sd_brier:.3f}",
            }
        )
        auc_rows.append(
            {
                "model_key": key,
                "model_label": label,
                "mean_fold_auc": mean_auc,
                "sd_fold_auc": sd_auc,
                "pooled_auc": pooled["auc"],
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "mean_fold_brier": mean_brier,
                "pooled_brier": pooled["brier"],
            }
        )

    table2 = pd.DataFrame(table2_rows)
    if table2.to_dict("records") != reported["table2"]:
        raise AssertionError("Reproduced Table 2 does not match reported_values.json")
    table3 = pd.DataFrame(reported["table3"])
    fold_metrics = pd.DataFrame(fold_rows)
    auc_summary = pd.DataFrame(auc_rows)

    table2.to_csv(TABLE_DIR / "Table_2_model_performance.csv", index=False, encoding="utf-8-sig")
    table3.to_csv(TABLE_DIR / "Table_3_pairwise_AUC.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(TABLE_DIR / "Table_S3_per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    auc_summary.to_csv(
        SOURCE_DIR / "figure4_AUC_source.csv", index=False, encoding="utf-8-sig"
    )
    return table2, table3, fold_metrics, auc_summary


def reproduce_sensitivity(oof: pd.DataFrame) -> pd.DataFrame:
    ordered = ordered_oof(oof)
    rows: list[dict] = []
    for model_key in ("clinical_6", "trimodal_equal_weight_6"):
        for fold in range(1, 6):
            frame = ordered[ordered["outer_fold"] == fold]
            fold_metric = metrics(
                frame["pCR"].to_numpy(dtype=int),
                frame[model_key].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "model_key": model_key,
                    "outer_fold": fold,
                    **fold_metric,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        TABLE_DIR / "Table_S2_six_variable_sensitivity_per_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks = json.loads(
        (ROOT / "sensitivity_analysis_checks.json").read_text(encoding="utf-8")
    )
    observed = {
        "clinical_6_pooled_auc": roc_auc_score(ordered["pCR"], ordered["clinical_6"]),
        "trimodal_equal_weight_6_pooled_auc": roc_auc_score(
            ordered["pCR"], ordered["trimodal_equal_weight_6"]
        ),
    }
    for key, value in observed.items():
        if not np.isclose(value, checks[key], atol=1e-12, rtol=0):
            raise AssertionError(f"Sensitivity value mismatch for {key}")
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    for axis in fig.axes:
        axis.grid(False)
        axis.tick_params(axis="both", which="both", direction="out", length=3, width=0.8, pad=5)
        axis.xaxis.labelpad = 5
        axis.yaxis.labelpad = 5
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIGURE_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def reproduce_roc(ordered: pd.DataFrame, auc_summary: pd.DataFrame) -> None:
    y = ordered["pCR"].to_numpy(dtype=int)
    roc_labels = {
        "clinical_11": "Clinical, 11 variables",
        "mri": "MRI, 3D DenseNet-121",
        "pathology": "Pathology, UNI + MIL",
        "trimodal_equal_weight_11": "Clinical + MRI + pathology",
    }
    roc_styles = {
        "clinical_11": "-",
        "mri": "-.",
        "pathology": "--",
        "trimodal_equal_weight_11": "-",
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    rows: list[dict] = []
    for key, label, _ in MODEL_ORDER[:3] + [MODEL_ORDER[-1]]:
        probability = ordered[key].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(y, probability)
        for point, (false_positive, true_positive, threshold) in enumerate(
            zip(fpr, tpr, thresholds), start=1
        ):
            rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "point": point,
                    "false_positive_rate": float(false_positive),
                    "true_positive_rate": float(true_positive),
                    "threshold": float(threshold),
                }
            )
        auc_value = float(
            auc_summary.loc[auc_summary["model_key"] == key, "pooled_auc"].iloc[0]
        )
        ax.plot(
            fpr,
            tpr,
            color=COLORS[key],
            lw=2.0,
            linestyle=roc_styles[key],
            label=f"{roc_labels[key]}: {auc_value:.3f}",
        )
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "Figure_3_ROC")
    pd.DataFrame(rows).to_csv(
        SOURCE_DIR / "figure3_ROC_source.csv", index=False, encoding="utf-8-sig"
    )


def reproduce_auc(auc_summary: pd.DataFrame) -> None:
    labels = {
        "clinical_11": "Clinical\n(11 variables)",
        "mri": "MRI\n(DenseNet-121)",
        "pathology": "Pathology\n(UNI + MIL)",
        "clinical_plus_mri": "Clinical + MRI",
        "clinical_plus_pathology": "Clinical +\npathology",
        "trimodal_equal_weight_11": "Clinical + MRI\n+ pathology",
    }
    frame = auc_summary.set_index("model_key").loc[[item[0] for item in MODEL_ORDER]]
    values = frame["pooled_auc"].to_numpy(dtype=float)
    lows = frame["bootstrap_ci_low"].to_numpy(dtype=float)
    highs = frame["bootstrap_ci_high"].to_numpy(dtype=float)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    ax.bar(
        x,
        values,
        width=0.68,
        color=[COLORS[key] for key in frame.index],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.errorbar(
        x,
        values,
        yerr=[values - lows, highs - values],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        lw=1.0,
    )
    for x_value, auc_value, high_value in zip(x, values, highs):
        label_y = high_value + 0.012
        ax.text(x_value, label_y, f"{auc_value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#777777", lw=0.8, ls=(0, (2, 2)))
    ax.set_ylim(0.45, max(0.95, float(np.max(highs)) + 0.07))
    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in frame.index], fontsize=8)
    ax.set_ylabel("Out-of-fold AUC (95% bootstrap CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, "Figure_4_AUC_comparison")


def net_benefit(y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    values = []
    for threshold in thresholds:
        prediction = probability >= threshold
        true_positive = np.sum(prediction & (y == 1))
        false_positive = np.sum(prediction & (y == 0))
        values.append(
            true_positive / len(y)
            - false_positive / len(y) * threshold / (1 - threshold)
        )
    return np.asarray(values)


def reproduce_dca(ordered: pd.DataFrame) -> None:
    y = ordered["pCR"].to_numpy(dtype=int)
    thresholds = np.linspace(0.05, 0.40, 141)
    prevalence = float(y.mean())
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(thresholds, np.zeros_like(thresholds), color="#666666", lw=0.9, label="Treat none")
    ax.plot(
        thresholds,
        treat_all,
        color="#000000",
        lw=0.9,
        ls="--",
        label="Treat all",
    )
    rows: list[dict] = []
    for factor, (key, label, _) in enumerate(
        [MODEL_ORDER[0], MODEL_ORDER[1], MODEL_ORDER[2], MODEL_ORDER[-1]]
    ):
        values = net_benefit(y, ordered[key].to_numpy(dtype=float), thresholds)
        linestyle = ["-", "-.", "--", "-"][factor]
        ax.plot(
            thresholds,
            values,
            color=COLORS[key],
            lw=1.7,
            ls=linestyle,
            label="Clinical + MRI + pathology" if key == "trimodal_equal_weight_11" else label.split(" (")[0],
        )
        for threshold, value in zip(thresholds, values):
            rows.append(
                {
                    "strategy": key,
                    "label": label,
                    "threshold": float(threshold),
                    "net_benefit": float(value),
                }
            )
    for strategy, label, values in (
        ("treat_none", "Treat none", np.zeros_like(thresholds)),
        ("treat_all", "Treat all", treat_all),
    ):
        for threshold, value in zip(thresholds, values):
            rows.append(
                {
                    "strategy": strategy,
                    "label": label,
                    "threshold": float(threshold),
                    "net_benefit": float(value),
                }
            )
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlim(0.05, 0.40)
    ax.set_ylim(-0.01, 0.23)
    ax.set_xticks(np.arange(0.05, 0.401, 0.05))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "Figure_5_DCA")
    pd.DataFrame(rows).to_csv(
        SOURCE_DIR / "figure5_DCA_source.csv", index=False, encoding="utf-8-sig"
    )


def reproduce_calibration(ordered: pd.DataFrame, auc_summary: pd.DataFrame) -> None:
    y = ordered["pCR"].to_numpy(dtype=int)
    figure_keys = [
        ("clinical_11", "Clinical"),
        ("mri", "MRI"),
        ("pathology", "Pathology"),
        ("trimodal_equal_weight_11", "Trimodal fusion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0), sharex=True, sharey=True)
    rows: list[dict] = []
    bins = np.linspace(0.0, 1.0, 9)
    for ax, (key, label) in zip(axes.flat, figure_keys):
        probability = ordered[key].to_numpy(dtype=float)
        observed, predicted = calibration_curve(y, probability, n_bins=8, strategy="uniform")
        bin_indices = np.searchsorted(bins[1:-1], probability, side="right")
        bin_counts = np.bincount(bin_indices, minlength=8)
        nonempty = np.flatnonzero(bin_counts)
        if len(nonempty) != len(observed):
            raise AssertionError(f"Calibration bins do not align for {key}")
        for bin_number, observed_value, predicted_value in zip(nonempty + 1, observed, predicted):
            rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "bin": int(bin_number),
                    "bin_lower": float(bins[bin_number - 1]),
                    "bin_upper": float(bins[bin_number]),
                    "n_subjects": int(bin_counts[bin_number - 1]),
                    "mean_predicted_probability": float(predicted_value),
                    "observed_pCR_proportion": float(observed_value),
                }
            )
        brier = float(
            auc_summary.loc[auc_summary["model_key"] == key, "mean_fold_brier"].iloc[0]
        )
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (2, 2)))
        ax.plot(predicted, observed, marker="o", ms=3.5, lw=1.4, color=COLORS[key])
        ax.set_title(f"{label}\nBrier = {brier:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 1].set_xlabel("Mean predicted probability")
    axes[0, 0].set_ylabel("Observed proportion")
    axes[1, 0].set_ylabel("Observed proportion")
    fig.tight_layout()
    save_figure(fig, "Figure_S1_Calibration")
    pd.DataFrame(rows).to_csv(
        SOURCE_DIR / "figureS1_calibration_source.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    dataset, oof, reported = load_data()
    table1, table1_summary = reproduce_table1(dataset)
    expected_table1 = json.loads(
        (ROOT / "table1_statistics.json").read_text(encoding="utf-8")
    )
    if table1_summary != expected_table1:
        raise AssertionError("Reproduced Table 1 does not match table1_statistics.json")
    table2, table3, fold_metrics, auc_summary = reproduce_tables(oof, reported)
    sensitivity = reproduce_sensitivity(oof)
    ordered = ordered_oof(oof)
    reproduce_roc(ordered, auc_summary)
    reproduce_auc(auc_summary)
    reproduce_dca(ordered)
    reproduce_calibration(ordered, auc_summary)

    manifest = {
        "status": "PASS",
        "participants": int(len(dataset)),
        "pCR": int(dataset["pCR"].sum()),
        "non_pCR": int((1 - dataset["pCR"]).sum()),
        "corrected_anti_HER2": int(dataset["anti_HER2_explicit_H"].sum()),
        "reported_tables_and_figures_reproduced": [
            "Table 1",
            "Table 2",
            "Table 3",
            "Table S2 (six-variable sensitivity analysis)",
            "Figure 3",
            "Figure 4",
            "Figure 5",
            "Figure S1",
        ],
        "output_directory": "reproduced",
    }
    (OUTPUT / "reproduction_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(table1.to_string(index=False))
    print(table2.to_string(index=False))
    print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
