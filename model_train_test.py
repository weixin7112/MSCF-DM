import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from feature_extraction import (
    ID_TO_SPECIES,
    SPECIES_CATALOG,
    SPECIES_TO_ID,
    DNADataset,
    prepare_test_data,
    prepare_training_data,
)
from model import DEFAULT_MODEL_CONFIG, build_model

METRIC_NAMES = ["ACC", "SN", "SP", "MCC", "AUC", "F1"]


def add_common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    train_parser.add_argument("--target-length", type=int, default=None)
    train_parser.add_argument("--n-splits", type=int, default=5)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--label-smoothing", type=float, default=0.02)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--ema-decay", type=float, default=0.999)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--disable-ema", action="store_true")
    add_common_runtime_arguments(train_parser)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--data-dir", type=Path, required=True)
    test_parser.add_argument("--weights-dir", type=Path, required=True)
    test_parser.add_argument("--output-dir", type=Path, default=Path("test_results"))
    test_parser.add_argument("--threshold", type=float, default=None)
    test_parser.add_argument("--deduplicate", action="store_true")
    add_common_runtime_arguments(test_parser)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "ACC": float(accuracy_score(labels, predictions)),
        "SN": float(sensitivity),
        "SP": float(specificity),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "AUC": float(auc),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def model_selection_key(metrics: dict) -> tuple:
    auc = metrics["AUC"] if np.isfinite(metrics["AUC"]) else -1.0
    return (
        round(float(metrics["MCC"]), 10),
        round(float(metrics["ACC"]), 10),
        round(float(auc), 10),
        round(float(metrics["F1"]), 10),
        -round(abs(float(metrics["SN"]) - float(metrics["SP"])), 10),
    )


def format_metrics(metrics: dict) -> str:
    return " ".join(
        [
            f"ACC={metrics['ACC']:.4f}",
            f"SN={metrics['SN']:.4f}",
            f"SP={metrics['SP']:.4f}",
            f"MCC={metrics['MCC']:.4f}",
            f"AUC={metrics['AUC']:.4f}",
            f"F1={metrics['F1']:.4f}",
        ]
    )


def smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            source_value = source_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)


def create_loader(
    dataset: DNADataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    indices: Optional[np.ndarray] = None,
) -> DataLoader:
    selected_dataset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    label_smoothing: float,
    gradient_clip: float,
    ema: Optional[ModelEMA],
    threshold: float,
) -> dict:
    model.train()
    total_loss = 0.0
    labels_all = []
    probabilities_all = []

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        k2_ids = batch["k2"].to(device, non_blocking=True)
        k3_ids = batch["k3"].to(device, non_blocking=True)
        species_ids = batch["species_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        smoothed_labels = smooth_labels(labels, label_smoothing)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, _, _ = model(tokens, k2_ids, k3_ids, species_ids)
            loss = criterion(logits, smoothed_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        total_loss += float(loss.detach().item()) * labels.size(0)
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(torch.sigmoid(logits.detach()).cpu().numpy())

    metrics = calculate_metrics(
        np.concatenate(labels_all),
        np.concatenate(probabilities_all),
        threshold,
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate_labeled(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_all = []
    probabilities_all = []
    indices_all = []
    gates_all = []

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        k2_ids = batch["k2"].to(device, non_blocking=True)
        k3_ids = batch["k3"].to(device, non_blocking=True)
        species_ids = batch["species_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, gate_summary, _ = model(tokens, k2_ids, k3_ids, species_ids)
            loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
        indices_all.append(batch["index"].numpy())
        gates_all.append(gate_summary.cpu().numpy())

    labels_array = np.concatenate(labels_all)
    probabilities_array = np.concatenate(probabilities_all)
    indices_array = np.concatenate(indices_all)
    gates_array = np.concatenate(gates_all)
    metrics = calculate_metrics(labels_array, probabilities_array, threshold)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, labels_array, probabilities_array, indices_array, gates_array


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities_all = []
    indices_all = []
    gates_all = []

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        k2_ids = batch["k2"].to(device, non_blocking=True)
        k3_ids = batch["k3"].to(device, non_blocking=True)
        species_ids = batch["species_id"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, gate_summary, _ = model(tokens, k2_ids, k3_ids, species_ids)

        probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
        indices_all.append(batch["index"].numpy())
        gates_all.append(gate_summary.cpu().numpy())

    return (
        np.concatenate(probabilities_all),
        np.concatenate(indices_all),
        np.concatenate(gates_all),
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_json_serializable_arguments(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def build_species_mapping_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "species_id": SPECIES_TO_ID[item["canonical"]],
                "canonical": item["canonical"],
                "scientific_name": item["scientific_name"],
                "aliases": ";".join(item["aliases"]),
            }
            for item in SPECIES_CATALOG
        ]
    )


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    use_ema = not args.disable_ema

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    dataframe, dataset, sequence_length, file_report = prepare_training_data(
        data_dir=args.data_dir,
        target_length=args.target_length,
        deduplicate=True,
        seed=args.seed,
        require_known_species=True,
    )
    labels = dataframe["label"].to_numpy(dtype=np.int64)
    species_ids = dataframe["species_id"].to_numpy(dtype=np.int64)

    if min(Counter(labels).values()) < args.n_splits:
        raise ValueError("The smallest class contains fewer samples than n_splits.")

    combined_strata = np.array(
        [f"{species_id}_{label}" for species_id, label in zip(species_ids, labels)]
    )
    strata_counts = Counter(combined_strata.tolist())
    strata = combined_strata if min(strata_counts.values()) >= args.n_splits else labels
    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(splitter.split(np.zeros(len(labels)), strata))

    np.savez(
        args.output_dir / "cv_splits.npz",
        **{
            f"fold_{fold}_{name}": values
            for fold, (train_indices, validation_indices) in enumerate(splits, start=1)
            for name, values in (
                ("train", train_indices),
                ("validation", validation_indices),
            )
        },
    )

    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    criterion = nn.BCEWithLogitsLoss()
    out_of_fold_probabilities = np.zeros(len(dataset), dtype=np.float64)
    out_of_fold_folds = np.zeros(len(dataset), dtype=np.int64)
    fold_results = []
    epoch_history = []
    gate_results = []

    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"Sequence length: {sequence_length}")
    print("Best checkpoint criterion: validation MCC")

    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        fold_start = time.time()
        seed_everything(args.seed + fold)
        train_loader = create_loader(
            dataset=dataset,
            indices=train_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            device=device,
        )
        validation_loader = create_loader(
            dataset=dataset,
            indices=validation_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            device=device,
        )

        model = build_model(
            sequence_length=sequence_length,
            num_species=len(SPECIES_TO_ID) + 1,
            **model_config,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = create_scheduler(optimizer, args.epochs)
        scaler = create_grad_scaler(amp_enabled)
        ema = ModelEMA(model, args.ema_decay) if use_ema else None

        best_key = None
        best_state = None
        best_epoch = 0
        best_metrics = None
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                label_smoothing=args.label_smoothing,
                gradient_clip=args.gradient_clip,
                ema=ema,
                threshold=args.threshold,
            )
            evaluation_model = ema.module if ema is not None else model
            validation_metrics, _, _, _, _ = evaluate_labeled(
                model=evaluation_model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                threshold=args.threshold,
            )
            scheduler.step()

            current_key = model_selection_key(validation_metrics)
            improved = best_key is None or current_key > best_key
            if improved:
                best_key = current_key
                best_epoch = epoch
                best_metrics = copy.deepcopy(validation_metrics)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in evaluation_model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Fold {fold}/{args.n_splits} Epoch {epoch:03d}/{args.epochs} "
                f"Train[{format_metrics(train_metrics)}] "
                f"Val[{format_metrics(validation_metrics)}] "
                f"lr={learning_rate:.2e} time={time.time() - epoch_start:.1f}s"
            )
            epoch_history.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    **{f"train_{name}": train_metrics[name] for name in METRIC_NAMES},
                    **{
                        f"validation_{name}": validation_metrics[name]
                        for name in METRIC_NAMES
                    },
                    "is_best": improved,
                }
            )

            if epochs_without_improvement >= args.patience:
                break

        if best_state is None or best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce a valid checkpoint.")

        model.load_state_dict(best_state)
        validation_metrics, _, probabilities, original_indices, gates = evaluate_labeled(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            threshold=args.threshold,
        )
        out_of_fold_probabilities[original_indices] = probabilities
        out_of_fold_folds[original_indices] = fold

        checkpoint = {
            "model_state_dict": best_state,
            "sequence_length": sequence_length,
            "model_config": model_config,
            "species_to_id": SPECIES_TO_ID,
            "fold": fold,
            "best_epoch": best_epoch,
            "selection_metric": "MCC",
            "decision_threshold": args.threshold,
            "weights_type": "EMA" if use_ema else "raw",
        }
        torch.save(checkpoint, weights_dir / f"fold_{fold}_best.pt")

        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                **{name: validation_metrics[name] for name in METRIC_NAMES},
                "TN": validation_metrics["TN"],
                "FP": validation_metrics["FP"],
                "FN": validation_metrics["FN"],
                "TP": validation_metrics["TP"],
                "minutes": (time.time() - fold_start) / 60.0,
            }
        )
        gate_results.append(
            {
                "fold": fold,
                "CNN": float(gates[:, 0].mean()),
                "BiGRU": float(gates[:, 1].mean()),
            }
        )
        print(
            f"Best fold {fold} epoch: {best_epoch}; "
            f"{format_metrics(validation_metrics)}"
        )

    fold_dataframe = pd.DataFrame(fold_results)
    history_dataframe = pd.DataFrame(epoch_history)
    gate_dataframe = pd.DataFrame(gate_results)
    oof_metrics = calculate_metrics(labels, out_of_fold_probabilities, args.threshold)

    oof_dataframe = dataframe.copy()
    oof_dataframe["fold"] = out_of_fold_folds
    oof_dataframe["probability"] = out_of_fold_probabilities
    oof_dataframe["prediction"] = (
        out_of_fold_probabilities >= args.threshold
    ).astype(np.int64)

    summary_dataframe = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": fold_dataframe[metric].mean(),
                "std": fold_dataframe[metric].std(ddof=1),
            }
            for metric in METRIC_NAMES
        ]
    )

    file_report.to_csv(args.output_dir / "file_report.csv", index=False)
    fold_dataframe.to_csv(args.output_dir / "fold_results.csv", index=False)
    history_dataframe.to_csv(args.output_dir / "epoch_history.csv", index=False)
    gate_dataframe.to_csv(args.output_dir / "gate_weights.csv", index=False)
    oof_dataframe.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    summary_dataframe.to_csv(args.output_dir / "cv_summary.csv", index=False)
    build_species_mapping_dataframe().to_csv(
        args.output_dir / "species_mapping.csv", index=False
    )

    configuration = {
        "sequence_length": sequence_length,
        "model_config": model_config,
        "training_config": make_json_serializable_arguments(args),
        "device": str(device),
        "species_to_id": SPECIES_TO_ID,
        "id_to_species": ID_TO_SPECIES,
        "fold_mean": {
            metric: float(fold_dataframe[metric].mean()) for metric in METRIC_NAMES
        },
        "fold_std": {
            metric: float(fold_dataframe[metric].std(ddof=1)) for metric in METRIC_NAMES
        },
        "oof_metrics": {
            metric: float(oof_metrics[metric]) for metric in METRIC_NAMES
        },
    }
    save_json(args.output_dir / "config.json", configuration)

    print("Cross-validation mean ± standard deviation")
    for metric in METRIC_NAMES:
        print(
            f"{metric}: {fold_dataframe[metric].mean():.4f} ± "
            f"{fold_dataframe[metric].std(ddof=1):.4f}"
        )
    print(f"OOF: {format_metrics(oof_metrics)}")


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"fold_(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def resolve_checkpoint_paths(weights_dir: Path) -> list[Path]:
    weights_dir = Path(weights_dir)
    if (weights_dir / "weights").is_dir():
        weights_dir = weights_dir / "weights"
    checkpoint_paths = sorted(weights_dir.glob("fold_*_best.pt"), key=checkpoint_sort_key)
    if not checkpoint_paths:
        checkpoint_paths = sorted(weights_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No model checkpoints were found in {weights_dir}.")
    return checkpoint_paths


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def validate_checkpoint_compatibility(checkpoints: list[dict]) -> None:
    first = checkpoints[0]
    required_keys = {
        "model_state_dict",
        "sequence_length",
        "model_config",
        "species_to_id",
    }
    missing = required_keys.difference(first)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")

    for checkpoint in checkpoints[1:]:
        if int(checkpoint["sequence_length"]) != int(first["sequence_length"]):
            raise ValueError("Checkpoints use different sequence lengths.")
        if checkpoint["model_config"] != first["model_config"]:
            raise ValueError("Checkpoints use different model configurations.")
        if checkpoint["species_to_id"] != first["species_to_id"]:
            raise ValueError("Checkpoints use different species mappings.")

    if first["species_to_id"] != SPECIES_TO_ID:
        raise ValueError(
            "The species mapping in the checkpoints differs from feature_extraction.py."
        )


def create_species_metric_table(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for species_id, group in dataframe.groupby("species_id", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        labels = group["label"].to_numpy(dtype=np.int64)
        metrics = calculate_metrics(labels, probabilities[indices], threshold)
        rows.append(
            {
                "species_id": int(species_id),
                "species": ID_TO_SPECIES.get(int(species_id), "UNKNOWN"),
                "samples": len(group),
                **{name: metrics[name] for name in METRIC_NAMES},
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
            }
        )
    return pd.DataFrame(rows)


def run_testing(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = resolve_checkpoint_paths(args.weights_dir)
    checkpoints = [load_checkpoint(path, device) for path in checkpoint_paths]
    validate_checkpoint_compatibility(checkpoints)

    sequence_length = int(checkpoints[0]["sequence_length"])
    model_config = checkpoints[0]["model_config"]
    checkpoint_thresholds = [
        float(checkpoint.get("decision_threshold", 0.5)) for checkpoint in checkpoints
    ]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.mean(checkpoint_thresholds))
    )

    dataframe, dataset, file_report, has_labels = prepare_test_data(
        data_dir=args.data_dir,
        target_length=sequence_length,
        deduplicate=args.deduplicate,
        seed=args.seed,
        require_known_species=True,
    )
    loader = create_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )

    fold_probabilities = []
    fold_metrics = []
    fold_gate_rows = []

    print(f"Device: {device}")
    print(f"Test samples: {len(dataset)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Decision threshold: {threshold:.4f}")

    for checkpoint_path, checkpoint in zip(checkpoint_paths, checkpoints):
        model = build_model(
            sequence_length=sequence_length,
            num_species=len(SPECIES_TO_ID) + 1,
            **model_config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        probabilities, indices, gates = predict_loader(
            model=model,
            loader=loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        ordered_probabilities = np.zeros(len(dataset), dtype=np.float64)
        ordered_gates = np.zeros((len(dataset), 2), dtype=np.float64)
        ordered_probabilities[indices] = probabilities
        ordered_gates[indices] = gates
        fold_probabilities.append(ordered_probabilities)

        fold_number = int(checkpoint.get("fold", checkpoint_sort_key(checkpoint_path)[0]))
        fold_gate_rows.append(
            {
                "fold": fold_number,
                "CNN": float(ordered_gates[:, 0].mean()),
                "BiGRU": float(ordered_gates[:, 1].mean()),
            }
        )
        if has_labels:
            labels = dataframe["label"].to_numpy(dtype=np.int64)
            metrics = calculate_metrics(labels, ordered_probabilities, threshold)
            fold_metrics.append(
                {
                    "fold": fold_number,
                    **{name: metrics[name] for name in METRIC_NAMES},
                    "TN": metrics["TN"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                    "TP": metrics["TP"],
                }
            )
            print(f"Fold {fold_number}: {format_metrics(metrics)}")
        else:
            print(f"Fold {fold_number}: prediction completed")

    probability_matrix = np.column_stack(fold_probabilities)
    ensemble_probabilities = probability_matrix.mean(axis=1)
    predictions = (ensemble_probabilities >= threshold).astype(np.int64)

    prediction_dataframe = dataframe.copy()
    for column_index, checkpoint in enumerate(checkpoints):
        fold_number = int(checkpoint.get("fold", column_index + 1))
        prediction_dataframe[f"fold_{fold_number}_probability"] = probability_matrix[
            :, column_index
        ]
    prediction_dataframe["ensemble_probability"] = ensemble_probabilities
    prediction_dataframe["prediction"] = predictions
    prediction_dataframe.to_csv(args.output_dir / "test_predictions.csv", index=False)
    file_report.to_csv(args.output_dir / "test_file_report.csv", index=False)
    pd.DataFrame(fold_gate_rows).to_csv(
        args.output_dir / "test_gate_weights.csv", index=False
    )

    summary = {
        "samples": len(dataset),
        "sequence_length": sequence_length,
        "checkpoint_count": len(checkpoints),
        "decision_threshold": threshold,
        "labeled_test_set": has_labels,
        "checkpoint_files": [path.name for path in checkpoint_paths],
    }

    if has_labels:
        labels = dataframe["label"].to_numpy(dtype=np.int64)
        ensemble_metrics = calculate_metrics(labels, ensemble_probabilities, threshold)
        pd.DataFrame(fold_metrics).to_csv(
            args.output_dir / "test_fold_metrics.csv", index=False
        )
        create_species_metric_table(
            dataframe=dataframe,
            probabilities=ensemble_probabilities,
            threshold=threshold,
        ).to_csv(args.output_dir / "test_species_metrics.csv", index=False)
        summary["ensemble_metrics"] = {
            name: float(ensemble_metrics[name]) for name in METRIC_NAMES
        }
        summary["confusion_matrix"] = {
            name: int(ensemble_metrics[name]) for name in ["TN", "FP", "FN", "TP"]
        }
        print(f"Ensemble: {format_metrics(ensemble_metrics)}")

    save_json(args.output_dir / "test_summary.json", summary)


if __name__ == "__main__":
    parsed_args = parse_arguments()
    if parsed_args.command == "train":
        run_training(parsed_args)
    else:
        run_testing(parsed_args)
