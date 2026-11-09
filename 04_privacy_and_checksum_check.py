"""Validate privacy, cohort alignment, prediction completeness, and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "checksums.sha256"
VALIDATION_REPORT = ROOT / "validation_report.json"
PRIVACY_REPORT = ROOT / "privacy_check_report.json"
EXTENDED_DIR = ROOT / "extended_visualizations"
REQUIRED_FILES = [
    "README.md",
    "DATA_DICTIONARY.csv",
    "SUGGESTED_VISUALIZATIONS.md",
    "requirements.txt",
    "analysis_dataset_deidentified.csv",
    "outer_fold_assignments.csv",
    "out_of_fold_predictions.csv",
    "table1_baseline_source.csv",
    "table1_statistics.json",
    "model_performance.csv",
    "per_fold_metrics.csv",
    "pairwise_auc_comparisons.csv",
    "sensitivity_analysis_summary.csv",
    "sensitivity_analysis_per_fold.csv",
    "figure3_ROC_source.csv",
    "figure4_AUC_source.csv",
    "figure5_DCA_source.csv",
    "figureS1_calibration_source.csv",
    "privacy_check_report.json",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_data() -> dict:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    dataset = pd.read_csv(ROOT / "analysis_dataset_deidentified.csv")
    folds = pd.read_csv(ROOT / "outer_fold_assignments.csv")
    oof = pd.read_csv(ROOT / "out_of_fold_predictions.csv")
    audit = pd.read_csv(ROOT / "her2_coding_audit.csv")

    forbidden_column_fragments = ("name", "姓名", "mrn", "medical_record", "id_card", "phone", "address")
    forbidden_columns = [
        column
        for column in dataset.columns
        if any(fragment in column.lower() for fragment in forbidden_column_fragments)
    ]
    if forbidden_columns:
        raise ValueError(f"Potential direct-identifier columns found: {forbidden_columns}")

    cjk = re.compile(r"[\u3400-\u9fff]")
    string_columns = dataset.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        hits = dataset[column].dropna().astype(str).map(lambda value: bool(cjk.search(value)))
        if hits.any():
            raise ValueError(f"CJK text found in de-identified column {column}")

    if len(dataset) != 105:
        raise ValueError("The de-identified dataset must contain 105 rows")
    if dataset["subject_id"].nunique() != 105:
        raise ValueError("Expected 105 unique de-identified participant IDs")
    if sorted(dataset["source_row"].tolist()) != list(range(1, 106)):
        raise ValueError("Source row sequence is incomplete")
    counts = {
        "participants": int(len(dataset)),
        "pCR": int(dataset["pCR"].sum()),
        "non_pCR": int((1 - dataset["pCR"]).sum()),
        "corrected_anti_HER2": int(dataset["anti_HER2_explicit_H"].sum()),
    }
    if counts != {
        "participants": 105,
        "pCR": 27,
        "non_pCR": 78,
        "corrected_anti_HER2": 39,
    }:
        raise ValueError(f"Unexpected cohort counts: {counts}")

    fold_rows = folds.groupby("subject_id")["split"].agg(
        total="size", validation=lambda values: int((values == "validation").sum())
    )
    if not (fold_rows["total"].eq(5).all() and fold_rows["validation"].eq(1).all()):
        raise ValueError("Each participant must have five rows and one validation row")
    fold_sizes = (
        folds[folds["split"] == "validation"]
        .groupby("outer_fold")
        .size()
        .reindex([1, 2, 3, 4, 5])
    )
    if not fold_sizes.eq(21).all():
        raise ValueError(f"Unexpected outer-fold sizes: {fold_sizes.to_dict()}")

    if len(oof) != 105 or oof["subject_id"].nunique() != 105:
        raise ValueError("OOF table must contain one row per participant")
    if oof["subject_id"].tolist() != dataset["subject_id"].tolist():
        raise ValueError("OOF subject order does not match the analysis dataset")
    if not np.array_equal(oof["pCR"].to_numpy(), dataset["pCR"].to_numpy()):
        raise ValueError("OOF outcomes do not align with the analysis dataset")
    probability_columns = [
        "clinical_11",
        "clinical_6",
        "mri",
        "pathology",
        "clinical_plus_mri",
        "clinical_plus_pathology",
        "trimodal_equal_weight_11",
        "trimodal_equal_weight_6",
    ]
    if oof[probability_columns].isna().any().any():
        raise ValueError("Missing OOF probabilities")
    probabilities = oof[probability_columns].to_numpy(dtype=float)
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("OOF probabilities fall outside [0, 1]")

    if int(audit["old_rule_contains_H_or_P"].sum()) != 68:
        raise ValueError("Old anti-HER2 audit count is not 68")
    if int(audit["corrected_rule_contains_explicit_H"].sum()) != 39:
        raise ValueError("Corrected anti-HER2 audit count is not 39")
    if int(audit["coding_changed"].sum()) != 29:
        raise ValueError("Expected 29 anti-HER2 coding changes")

    forbidden_source_files = [
        path.name
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
    ]
    if forbidden_source_files:
        raise ValueError(f"Source spreadsheets must not be included: {forbidden_source_files}")

    return {
        "status": "PASS",
        **counts,
        "outer_fold_sizes": [int(value) for value in fold_sizes],
        "old_anti_HER2_rule_count": 68,
        "records_changed_by_correction": 29,
        "oof_rows": int(len(oof)),
        "oof_probability_columns": probability_columns,
        "missing_oof_values": 0,
        "direct_identifier_columns": 0,
        "source_spreadsheets_in_package": 0,
    }


def validate_extended_visualizations() -> dict:
    expected_ids = [f"EV{index:02d}" for index in range(1, 19)]
    figure_dir = EXTENDED_DIR / "figures"
    source_dir = EXTENDED_DIR / "figure_source"
    index_path = EXTENDED_DIR / "VISUALIZATION_INDEX.csv"
    required_review_files = [
        EXTENDED_DIR / "VISUALIZATION_ATLAS.pdf",
        EXTENDED_DIR / "VISUALIZATION_CONTACT_SHEET.png",
        EXTENDED_DIR / "VISUALIZATION_README.md",
    ]
    if not index_path.exists():
        raise FileNotFoundError("Extended visualization index is missing")
    missing_review_files = [
        str(path.relative_to(ROOT)) for path in required_review_files if not path.exists()
    ]
    if missing_review_files:
        raise FileNotFoundError(f"Extended review files are missing: {missing_review_files}")

    index = pd.read_csv(index_path)
    if index["figure_id"].tolist() != expected_ids:
        raise ValueError("Extended visualization index must list EV01-EV18 in order")
    if not index["status"].eq("complete").all():
        raise ValueError("Every extended visualization must have complete status")

    cjk = re.compile(r"[\u3400-\u9fff]")
    absolute_path = re.compile(r"(?:[A-Za-z]:\\|\\\\[^\\s]+\\\\)")
    figure_counts = {"png": 0, "pdf": 0, "tiff": 0, "source_csv": 0}
    for figure_id in expected_ids:
        for suffix, key in (("png", "png"), ("pdf", "pdf"), ("tiff", "tiff")):
            matches = sorted(figure_dir.glob(f"{figure_id}_*.{suffix}"))
            if len(matches) != 1 or matches[0].stat().st_size == 0:
                raise ValueError(f"Expected one non-empty {suffix} file for {figure_id}")
            figure_counts[key] += 1
        source_matches = sorted(source_dir.glob(f"{figure_id}_*_source.csv"))
        if len(source_matches) != 1 or source_matches[0].stat().st_size == 0:
            raise ValueError(f"Expected one non-empty source CSV for {figure_id}")
        source_text = source_matches[0].read_text(encoding="utf-8-sig")
        if cjk.search(source_text):
            raise ValueError(f"CJK text found in public figure source: {source_matches[0].name}")
        if absolute_path.search(source_text):
            raise ValueError(f"Absolute path found in public figure source: {source_matches[0].name}")
        figure_counts["source_csv"] += 1

    forbidden_raw_suffixes = {".nii", ".gz", ".svs", ".jpeg", ".jpg", ".xlsx", ".xls"}
    forbidden_raw_files = [
        path.relative_to(ROOT).as_posix()
        for path in EXTENDED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_raw_suffixes
    ]
    if forbidden_raw_files:
        raise ValueError(f"Raw patient data found in public visualization directory: {forbidden_raw_files}")

    return {
        "status": "PASS",
        "figures": len(expected_ids),
        **figure_counts,
        "atlas_present": True,
        "contact_sheet_present": True,
        "public_source_cjk_hits": 0,
        "public_source_absolute_paths": 0,
        "raw_patient_files": 0,
    }


def package_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in {"checksums.sha256", "validation_report.json"}:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())


def write_checksums() -> None:
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in package_files()
    ]
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums() -> dict:
    if not CHECKSUM_FILE.exists():
        raise FileNotFoundError("checksums.sha256 has not been generated")
    expected = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, relative = line.split("  ", 1)
        expected[relative] = checksum
    actual_files = {
        path.relative_to(ROOT).as_posix(): path for path in package_files()
    }
    missing = sorted(set(expected) - set(actual_files))
    unexpected = sorted(set(actual_files) - set(expected))
    mismatches = [
        relative
        for relative, checksum in expected.items()
        if relative in actual_files and digest(actual_files[relative]) != checksum
    ]
    return {
        "files_checked": len(expected),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "checksum_mismatches": mismatches,
        "status": "PASS" if not missing and not unexpected and not mismatches else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    data_report = validate_data()
    extended_report = validate_extended_visualizations()
    privacy_report = {**data_report, "visualization_privacy": extended_report}
    PRIVACY_REPORT.write_text(
        json.dumps(privacy_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_checksums:
        write_checksums()
    checksum_report = verify_checksums()
    report = {
        "data_integrity": data_report,
        "extended_visualization_integrity": extended_report,
        "checksum_integrity": checksum_report,
    }
    VALIDATION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if checksum_report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
