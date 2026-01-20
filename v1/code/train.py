#!/usr/bin/env python3
"""ALS training script for v1.

This script trains the ALS edge classifier used in the paper on an edge
classification CSV, with two modes:

- single_stage: train all weights (semantic, emotional, temporal, bias) jointly.
- multistage:  stage 1 trains semantic + temporal + bias; stage 2 fine-tunes the
               emotional weight only, starting from stage 1's best checkpoint.

Configuration is provided via a JSON file (see training_config.json) and can be
optionally overridden on the command line.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import ALSModel


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_datasets(cfg: Dict) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    import pandas as pd

    data_csv = cfg["data_csv"]
    df = pd.read_csv(data_csv)

    feature_fields = cfg["feature_fields"]
    label_field = cfg["label_field"]

    X = torch.tensor(df[feature_fields].values, dtype=torch.float32)
    y = torch.tensor(np.expand_dims(df[label_field].values, axis=-1), dtype=torch.float32)

    dataset = TensorDataset(X, y)

    val_split = float(cfg.get("val_split", 0.15))
    test_split = float(cfg.get("test_split", 0.15))
    
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size

    generator = torch.Generator().manual_seed(int(cfg.get("random_seed", 42)))
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_ds, val_ds, test_ds


def evaluate_dataset(dataset: TensorDataset, model: ALSModel, pos_weight: float = 1.0) -> Dict[str, float]:
    """Compute metrics (Loss, Accuracy, Precision, Recall, F1) over a dataset."""

    loss_fn = torch.nn.BCELoss(reduction="none")
    loader = DataLoader(dataset, batch_size=256)
    model.eval()
    
    total_loss = 0.0
    n_samples = 0
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_prob = model(X_batch)
            raw_loss = loss_fn(y_prob, y_batch)
            
            # Apply pos_weight manually to match training objective
            weights = torch.ones_like(y_batch)
            weights[y_batch == 1] = pos_weight
            loss = (raw_loss * weights).mean()

            batch_size = y_batch.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            all_y_true.extend(y_batch.numpy().flatten())
            all_y_prob.extend(y_prob.numpy().flatten())
            all_y_pred.extend((y_prob > 0.5).float().numpy().flatten())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    avg_loss = total_loss / max(n_samples, 1)
    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def train_single_stage(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    model: ALSModel,
    cfg: Dict,
) -> Tuple[ALSModel, Dict]:
    """Train all ALS weights jointly using configuration hyperparameters."""

    num_epochs = int(cfg.get("num_epochs", 100))
    lr = float(cfg.get("learning_rate", 0.001))
    patience = int(cfg.get("early_stopping_patience", 10))
    batch_size = int(cfg.get("batch_size", 128))

    # Calculate pos_weight for imbalance
    y_train = torch.cat([y for _, y in train_ds], dim=0)
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    
    use_pos_weight = cfg.get("use_pos_weight", True)
    if use_pos_weight:
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"Calculated pos_weight: {pos_weight:.4f} (Neg: {n_neg}, Pos: {n_pos})")
    else:
        pos_weight = 1.0
        print("Using pos_weight: 1.0 (Disabled by config)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Use reduction='none' to apply manual weight
    loss_fn = torch.nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, threshold=1e-4
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    metrics = {"train_loss": [], "val_loss": [], "pos_weight": pos_weight}

    for epoch in range(num_epochs):
        model.train()
        running_train = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            raw_loss = loss_fn(y_pred, y_batch)
            
            # Apply pos_weight manually
            weights = torch.ones_like(y_batch)
            weights[y_batch == 1] = pos_weight
            loss = (raw_loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            batch_size_cur = y_batch.shape[0]
            running_train += loss.item() * batch_size_cur
            n_train += batch_size_cur

        avg_train = running_train / max(n_train, 1)

        # Validation
        model.eval()
        running_val = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                raw_v_loss = loss_fn(y_pred, y_batch)
                
                # Apply pos_weight manually
                weights = torch.ones_like(y_batch)
                weights[y_batch == 1] = pos_weight
                v_loss = (raw_v_loss * weights).mean()

                batch_size_cur = y_batch.shape[0]
                running_val += v_loss.item() * batch_size_cur
                n_val += batch_size_cur
        avg_val = running_val / max(n_val, 1)

        metrics["train_loss"].append(avg_train)
        metrics["val_loss"].append(avg_val)

        print(f"[single_stage] Epoch {epoch+1}/{num_epochs} - train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val)
        if epochs_no_improve >= patience:
            print("Early stopping (single_stage)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, metrics


def _apply_gradient_mask(model: ALSModel, mask_emotion: bool, mask_semantic: bool, mask_temporal: bool, mask_bias: bool) -> None:
    """Zero out gradients for specific weight components before optimizer.step()."""

    if model.slp.weight.grad is not None:
        # Weight layout: [ [w_semantic, w_emotional, w_temporal] ]
        if mask_semantic:
            model.slp.weight.grad[0, 0] = 0.0
        if mask_emotion:
            model.slp.weight.grad[0, 1] = 0.0
        if mask_temporal:
            model.slp.weight.grad[0, 2] = 0.0
    if mask_bias and model.slp.bias.grad is not None:
        model.slp.bias.grad[0] = 0.0


def train_multistage(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    model: ALSModel,
    cfg: Dict,
) -> Tuple[ALSModel, Dict]:
    """Two-stage training: (1) semantic+temporal, (2) emotional only.

    Stage 1: optimise semantic + temporal weights and bias (emotion weight frozen).
    Stage 2: starting from best Stage 1, optimise emotional weight only (others frozen).
    """

    lr = float(cfg.get("learning_rate", 0.001))
    patience = int(cfg.get("early_stopping_patience", 10))
    batch_size = int(cfg.get("batch_size", 128))
    stage1_epochs = int(cfg.get("stage1_epochs", 50))
    stage2_epochs = int(cfg.get("stage2_epochs", 50))

    # Calculate pos_weight for imbalance
    y_train = torch.cat([y for _, y in train_ds], dim=0)
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    
    use_pos_weight = cfg.get("use_pos_weight", True)
    if use_pos_weight:
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"Calculated pos_weight: {pos_weight:.4f} (Neg: {n_neg}, Pos: {n_pos})")
    else:
        pos_weight = 1.0
        print("Using pos_weight: 1.0 (Disabled by config)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Use reduction='none' for manual weighting
    loss_fn = torch.nn.BCELoss(reduction="none")

    metrics = {
        "stage1_train_loss": [],
        "stage1_val_loss": [],
        "stage2_train_loss": [],
        "stage2_val_loss": [],
        "pos_weight": pos_weight,
    }

    # -------------------------
    # Stage 1: semantic + temporal + bias
    # -------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, threshold=1e-4
    )

    best_val_loss = float("inf")
    best_state_stage1 = None
    epochs_no_improve = 0

    print("=== Multistage training: Stage 1 (semantic + temporal + bias) ===")
    for epoch in range(stage1_epochs):
        model.train()
        running_train = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            raw_loss = loss_fn(y_pred, y_batch)
            
            # Apply pos_weight manually
            weights = torch.ones_like(y_batch)
            weights[y_batch == 1] = pos_weight
            loss = (raw_loss * weights).mean()

            loss.backward()
            # Freeze emotion weight (index 1) during Stage 1
            _apply_gradient_mask(
                model,
                mask_emotion=True,
                mask_semantic=False,
                mask_temporal=False,
                mask_bias=False,
            )
            optimizer.step()
            batch_size_cur = y_batch.shape[0]
            running_train += loss.item() * batch_size_cur
            n_train += batch_size_cur

        avg_train = running_train / max(n_train, 1)

        # Validation
        model.eval()
        running_val = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                raw_v_loss = loss_fn(y_pred, y_batch)
                
                # Apply pos_weight manually
                weights = torch.ones_like(y_batch)
                weights[y_batch == 1] = pos_weight
                v_loss = (raw_v_loss * weights).mean()

                batch_size_cur = y_batch.shape[0]
                running_val += v_loss.item() * batch_size_cur
                n_val += batch_size_cur
        avg_val = running_val / max(n_val, 1)

        metrics["stage1_train_loss"].append(avg_train)
        metrics["stage1_val_loss"].append(avg_val)

        print(f"[multistage:stage1] Epoch {epoch+1}/{stage1_epochs} - train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state_stage1 = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val)
        if epochs_no_improve >= patience:
            print("Early stopping (multistage stage 1)")
            break

    if best_state_stage1 is not None:
        model.load_state_dict(best_state_stage1)

    # -------------------------
    # Stage 2: emotional weight only
    # -------------------------
    print("=== Multistage training: Stage 2 (emotional weight only) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, threshold=1e-4
    )

    best_val_loss_stage2 = float("inf")
    best_state_stage2 = None
    epochs_no_improve = 0

    for epoch in range(stage2_epochs):
        model.train()
        running_train = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            raw_loss = loss_fn(y_pred, y_batch)

            # Apply pos_weight manually
            weights = torch.ones_like(y_batch)
            weights[y_batch == 1] = pos_weight
            loss = (raw_loss * weights).mean()

            loss.backward()
            # Only allow emotional weight (index 1) to update in Stage 2
            _apply_gradient_mask(
                model,
                mask_emotion=False,
                mask_semantic=True,
                mask_temporal=True,
                mask_bias=True,
            )
            optimizer.step()
            batch_size_cur = y_batch.shape[0]
            running_train += loss.item() * batch_size_cur
            n_train += batch_size_cur

        avg_train = running_train / max(n_train, 1)

        # Validation
        model.eval()
        running_val = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                raw_v_loss = loss_fn(y_pred, y_batch)

                # Apply pos_weight manually
                weights = torch.ones_like(y_batch)
                weights[y_batch == 1] = pos_weight
                v_loss = (raw_v_loss * weights).mean()

                batch_size_cur = y_batch.shape[0]
                running_val += v_loss.item() * batch_size_cur
                n_val += batch_size_cur
        avg_val = running_val / max(n_val, 1)

        metrics["stage2_train_loss"].append(avg_train)
        metrics["stage2_val_loss"].append(avg_val)

        print(f"[multistage:stage2] Epoch {epoch+1}/{stage2_epochs} - train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss_stage2:
            best_val_loss_stage2 = avg_val
            best_state_stage2 = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val)
        if epochs_no_improve >= patience:
            print("Early stopping (multistage stage 2)")
            break

    if best_state_stage2 is not None:
        model.load_state_dict(best_state_stage2)

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALS edge classifier for v1 experiments.")
    parser.add_argument(
        "--config",
        default="v1/code/training_config.json",
        help="Path to training configuration JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_stage", "multistage"],
        default=None,
        help="Override training mode; defaults to value in config.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    mode = args.mode or cfg.get("mode", "single_stage")
    if mode not in {"single_stage", "multistage"}:
        raise ValueError(f"Invalid mode: {mode}")

    # Ensure output directory exists
    out_path = Path(cfg.get("output_model_path", "v1/artifacts/pretrained/als.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {cfg['data_csv']}")
    train_ds, val_ds, test_ds = make_datasets(cfg)

    input_dim = len(cfg.get("feature_fields", []))
    model = ALSModel(input_dim=input_dim, ignore_feature=None)
    
    # Baseline check
    baseline_metrics = evaluate_dataset(val_ds, model)
    print(f"Baseline validation BCE loss (untrained model): {baseline_metrics['loss']:.4f}")

    if mode == "single_stage":
        print("Running single-stage training (all weights jointly)...")
        model, metrics = train_single_stage(train_ds, val_ds, model, cfg)
    else:
        print("Running multistage training (semantic+time, then emotion)...")
        model, metrics = train_multistage(train_ds, val_ds, model, cfg)

    # Final Evaluation
    pos_weight = metrics.get("pos_weight", 1.0)
    val_results = evaluate_dataset(val_ds, model, pos_weight)
    test_results = evaluate_dataset(test_ds, model, pos_weight)

    print(f"\nFinal Validation Results:")
    for m, v in val_results.items():
        print(f"  {m:10}: {v:.4f}")

    print(f"\nFinal Test Results:")
    for m, v in test_results.items():
        print(f"  {m:10}: {v:.4f}")

    # Save test results to metrics
    metrics["test_performance"] = test_results

    # Print observed weights
    weights = model.slp.weight.detach().numpy()[0]
    bias = model.slp.bias.detach().numpy()[0]

    # Save weight values to metrics
    feature_names = cfg.get("feature_fields", [])
    metrics["final_weights"] = {name: float(w) for name, w in zip(feature_names, weights)}
    metrics["final_weights"]["Bias"] = float(bias)

    print("\n--- Observed Weights ---")
    for name, val in metrics["final_weights"].items():
        print(f"{name:20}: {val:.4f}")
    print("------------------------\n")

    # Save results (metrics, weights, and config)
    results = {
        "config": cfg,
        "metrics": metrics
    }
    metrics_path = out_path.parent / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to {metrics_path}")

    torch.save(model.state_dict(), out_path)
    print(f"Trained model saved to {out_path}")


if __name__ == "__main__":
    main()
