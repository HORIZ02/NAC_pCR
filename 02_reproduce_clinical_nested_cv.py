"""Retrain the clinical nested cross-validation from the de-identified dataset.

This optional script reproduces the clinical workflow. The archived clinical OOF
predictions remain the reference because GPU and library-level nondeterminism can
change the last decimals after retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduced" / "clinical_retraining"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
MAX_EPOCHS = 120
PATIENCE = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FULL_FEATURES = [
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
REDUCED_FEATURES = [
    "ER_percent",
    "PR_percent",
    "Ki67_percent",
    "HER2_IHC",
    "clinical_N_stage",
    "anti_HER2_explicit_H",
]


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: torch.Tensor):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        cross_entropy = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        return (((1 - probability) ** self.gamma) * cross_entropy).mean()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_features(frame: pd.DataFrame, full: bool) -> np.ndarray:
    table = pd.DataFrame(
        {
            "Age_years": frame["Age_years"].astype(float) / 100.0,
            "ER_percent": frame["ER_percent"].astype(float) / 100.0,
            "PR_percent": frame["PR_percent"].astype(float) / 100.0,
            "Ki67_percent": frame["Ki67_percent"].astype(float) / 100.0,
            "HER2_IHC": frame["HER2_IHC"].astype(float) / 3.0,
            "menopausal_status_0_1": frame["menopausal_status_0_1"].astype(float),
            "histological_grade": (frame["histological_grade"].astype(float) - 1.0) / 2.0,
            "clinical_T_stage": (frame["clinical_T_stage"].astype(float) - 1.0) / 3.0,
            "clinical_N_stage": frame["clinical_N_stage"].astype(float) / 3.0,
            "V1_log": frame["V1_log"].astype(float),
            "anti_HER2_explicit_H": frame["anti_HER2_explicit_H"].astype(float),
        }
    )
    columns = FULL_FEATURES if full else REDUCED_FEATURES
    return table[columns].to_numpy(dtype=np.float32)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict_scaled(
    scaler: StandardScaler, model: ClinicalMLP, features: np.ndarray, device: torch.device
) -> np.ndarray:
    scaled = scaler.transform(features).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(scaled, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def train_nested(
    frame: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    fold_assignments: pd.DataFrame,
    model_name: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    subject_to_index = {
        subject_id: index for index, subject_id in enumerate(frame["subject_id"].tolist())
    }
    fold_results: list[dict] = []
    pooled_y: list[int] = []
    pooled_probability: list[float] = []

    for fold_idx in range(OUTER_FOLDS):
        fold_frame = fold_assignments[fold_assignments["outer_fold"] == fold_idx + 1]
        validation_ids = fold_frame.loc[fold_frame["split"] == "validation", "subject_id"]
        training_ids = fold_frame.loc[fold_frame["split"] == "training", "subject_id"]
        validation_indices = np.asarray(
            sorted(subject_to_index[subject_id] for subject_id in validation_ids), dtype=int
        )
        training_indices = np.asarray(
            sorted(subject_to_index[subject_id] for subject_id in training_ids), dtype=int
        )

        x_train_outer = features[training_indices]
        y_train_outer = labels[training_indices]
        x_validation_outer = features[validation_indices]
        y_validation_outer = labels[validation_indices]

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold_idx
        )
        inner_auc_curves: list[list[float]] = []
        for inner_idx, (inner_train, inner_validation) in enumerate(
            inner.split(x_train_outer, y_train_outer)
        ):
            scaler = StandardScaler()
            x_inner_train = scaler.fit_transform(x_train_outer[inner_train]).astype(np.float32)
            x_inner_validation = scaler.transform(
                x_train_outer[inner_validation]
            ).astype(np.float32)
            set_seed(1000 + fold_idx * 10 + inner_idx)
            model = ClinicalMLP(x_inner_train.shape[1]).to(device)
            criterion = FocalLoss(
                2.0, class_weights(y_train_outer[inner_train], device)
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS
            )
            input_train = torch.tensor(x_inner_train, dtype=torch.float32, device=device)
            target_train = torch.tensor(
                y_train_outer[inner_train], dtype=torch.long, device=device
            )
            input_validation = torch.tensor(
                x_inner_validation, dtype=torch.float32, device=device
            )
            target_validation = y_train_outer[inner_validation]
            curve: list[float] = []
            for _ in range(max_epochs):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(input_train), target_train)
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.eval()
                with torch.no_grad():
                    probability = (
                        torch.softmax(model(input_validation), dim=1)[:, 1].cpu().numpy()
                    )
                curve.append(float(roc_auc_score(target_validation, probability)))
            inner_auc_curves.append(curve)

        minimum_length = min(len(curve) for curve in inner_auc_curves)
        mean_curve = np.mean(
            [curve[:minimum_length] for curve in inner_auc_curves], axis=0
        )
        best_epoch = int(np.argmax(mean_curve) + 1)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_outer).astype(np.float32)
        model = ClinicalMLP(x_train_scaled.shape[1]).to(device)
        criterion = FocalLoss(2.0, class_weights(y_train_outer, device))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )
        input_train = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
        target_train = torch.tensor(y_train_outer, dtype=torch.long, device=device)
        for _ in range(best_epoch):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(input_train), target_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

        probability = predict_scaled(
            scaler, model, x_validation_outer, device
        )
        prediction = (probability >= 0.5).astype(int)
        result = {
            "fold": fold_idx,
            "train_n": int(len(training_indices)),
            "validation_n": int(len(validation_indices)),
            "best_epoch_inner_auc": best_epoch,
            "inner_best_auc": float(mean_curve[best_epoch - 1]),
            "auc": float(roc_auc_score(y_validation_outer, probability)),
            "accuracy": float(accuracy_score(y_validation_outer, prediction)),
            "sensitivity": float(
                recall_score(y_validation_outer, prediction, zero_division=0)
            ),
            "specificity": float(
                recall_score(
                    y_validation_outer, prediction, pos_label=0, zero_division=0
                )
            ),
            "brier": float(brier_score_loss(y_validation_outer, probability)),
            "log_loss": float(log_loss(y_validation_outer, probability, labels=[0, 1])),
            "y_true": y_validation_outer.tolist(),
            "y_prob": probability.tolist(),
            "patient_indices": validation_indices.tolist(),
        }
        fold_results.append(result)
        pooled_y.extend(y_validation_outer.tolist())
        pooled_probability.extend(probability.tolist())
        print(
            f"{model_name} fold {fold_idx + 1}: "
            f"AUC={result['auc']:.4f}, epoch={best_epoch}"
        )

    return (
        fold_results,
        np.asarray(pooled_y, dtype=int),
        np.asarray(pooled_probability, dtype=float),
    )


def summarize(folds: list[dict], y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "mean_fold_auc": float(np.mean([fold["auc"] for fold in folds])),
        "sd_fold_auc": float(np.std([fold["auc"] for fold in folds], ddof=1)),
        "pooled_auc": float(roc_auc_score(y, probability)),
        "mean_fold_brier": float(np.mean([fold["brier"] for fold in folds])),
        "sd_fold_brier": float(np.std([fold["brier"] for fold in folds], ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.set_num_threads(1)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    set_seed(SEED)

    frame = pd.read_csv(ROOT / "analysis_dataset_deidentified.csv")
    assignments = pd.read_csv(ROOT / "outer_fold_assignments.csv")
    labels = frame["pCR"].to_numpy(dtype=int)
    result: dict = {
        "device": str(device),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "note": (
            "Retraining is provided for methodological reproducibility. "
            "The archived OOF predictions are the reference values for the paper."
        ),
        "feature_sets": {"full_11": FULL_FEATURES, "reduced_6": REDUCED_FEATURES},
    }

    archived = pd.read_csv(ROOT / "out_of_fold_predictions.csv").set_index("subject_id")
    for name, features in (
        ("full_11", build_features(frame, full=True)),
        ("reduced_6", build_features(frame, full=False)),
    ):
        folds, y, probability = train_nested(
            frame,
            features,
            labels,
            assignments,
            name,
            device,
            args.max_epochs,
            args.patience,
        )
        result[name] = {
            "summary": summarize(folds, y, probability),
            "per_fold": folds,
            "pooled_y": y.tolist(),
            "pooled_probability": probability.tolist(),
        }
        reference_column = "clinical_11" if name == "full_11" else "clinical_6"
        reference_ids = [
            f"P{int(index) + 1:03d}"
            for fold in folds
            for index in fold["patient_indices"]
        ]
        reference_probability = archived.loc[reference_ids, reference_column].to_numpy()
        result[name][f"comparison_with_archived_{reference_column}"] = {
            "mean_absolute_difference": float(
                np.mean(np.abs(probability - reference_probability))
            ),
            "max_absolute_difference": float(
                np.max(np.abs(probability - reference_probability))
            ),
            "auc_retrained": float(roc_auc_score(y, probability)),
            "auc_archived": float(roc_auc_score(y, reference_probability)),
        }

    (OUTPUT / "clinical_nested_cv_rerun.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: result[key]["summary"] for key in ("full_11", "reduced_6")}, indent=2))


if __name__ == "__main__":
    main()
