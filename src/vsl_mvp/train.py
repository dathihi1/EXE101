from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .models import build_model
from .utils import seed_everything, write_json


class AugmentedSequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment_copies: int = 0,
        noise_std: float = 0.015,
        scale_std: float = 0.035,
        frame_drop_prob: float = 0.04,
        max_time_shift: int = 5,
        min_time_scale: float = 0.9,
        max_time_scale: float = 1.12,
    ):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment_copies = max(0, int(augment_copies))
        self.noise_std = noise_std
        self.scale_std = scale_std
        self.frame_drop_prob = frame_drop_prob
        self.max_time_shift = max_time_shift
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale

    def __len__(self) -> int:
        return len(self.y) * (1 + self.augment_copies)

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.y)
        x = self.X[base_idx].clone()
        if idx >= len(self.y):
            x = self._augment(x)
        return x, self.y[base_idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_scale(x)
        x = self._time_shift(x)
        nonzero = x.ne(0.0)
        if self.scale_std > 0:
            x = x * torch.clamp(1.0 + torch.randn(1) * self.scale_std, 0.88, 1.12)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * nonzero
        if self.frame_drop_prob > 0:
            drop = torch.rand(x.shape[0], device=x.device) < self.frame_drop_prob
            x[drop] = 0.0
        return x

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_time_shift <= 0:
            return x
        shift = int(torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,)).item())
        if shift == 0:
            return x
        out = torch.zeros_like(x)
        if shift > 0:
            out[shift:] = x[:-shift]
        else:
            out[:shift] = x[-shift:]
        return out

    def _time_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_time_scale <= 0 or self.max_time_scale <= 0 or self.min_time_scale == self.max_time_scale:
            return x
        length = x.shape[0]
        scale = float(torch.empty(1).uniform_(self.min_time_scale, self.max_time_scale).item())
        scaled_len = max(4, int(round(length * scale)))
        scaled = F.interpolate(
            x.T.unsqueeze(0),
            size=scaled_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T
        if scaled_len == length:
            return scaled
        if scaled_len > length:
            start = int(torch.randint(0, scaled_len - length + 1, (1,)).item())
            return scaled[start : start + length]
        out = torch.zeros_like(x)
        start = int(torch.randint(0, length - scaled_len + 1, (1,)).item())
        out[start : start + scaled_len] = scaled
        return out


def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean([target in row for target, row in zip(y, topk)]))


def make_split(y: np.ndarray, signers: np.ndarray, test_size: float, seed: int):
    usable_groups = np.asarray([s for s in signers])
    has_groups = len(set(usable_groups.tolist()) - {""}) >= 3
    if has_groups:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros_like(y), y, groups=usable_groups))
        return train_idx, val_idx, "group_shuffle_by_signer"
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y if min(np.bincount(y)) >= 2 else None,
    )
    return train_idx, val_idx, "stratified_random"


def evaluate(model, loader, device):
    model.eval()
    logits_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.numpy())
    logits_np = np.concatenate(logits_all)
    y_np = np.concatenate(y_all)
    pred = logits_np.argmax(axis=1)
    return {
        "top1": float(accuracy_score(y_np, pred)),
        "top3": topk_accuracy(logits_np, y_np, min(3, logits_np.shape[1])),
        "macro_f1": float(f1_score(y_np, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_np, pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VSL MVP sequence classifier.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model", choices=["gru", "transformer", "lite_transformer"], default="transformer")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=35, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--augment-copies", default=3, type=int, help="Synthetic augmented copies per training sample.")
    parser.add_argument("--noise-std", default=0.015, type=float)
    parser.add_argument("--scale-std", default=0.035, type=float)
    parser.add_argument("--frame-drop-prob", default=0.04, type=float)
    parser.add_argument("--max-time-shift", default=5, type=int)
    parser.add_argument("--min-time-scale", default=0.9, type=float)
    parser.add_argument("--max-time-scale", default=1.12, type=float)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--confidence-threshold", default=0.55, type=float)
    args = parser.parse_args()

    seed_everything(args.seed)
    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    labels = [str(x) for x in data["labels"].tolist()]
    signers = data["signers"].astype(str) if "signers" in data.files else np.asarray([""] * len(y))
    statuses = data["statuses"].astype(str) if "statuses" in data.files else np.asarray(["ok"] * len(y))
    schema_version = str(data["schema_version"][0]) if "schema_version" in data.files else "v1_hands_pose"
    feature_metadata = {}
    if "schema_metadata" in data.files:
        try:
            feature_metadata = data["schema_metadata"][0].item()
        except AttributeError:
            feature_metadata = data["schema_metadata"][0]
    keep = statuses == "ok"
    if keep.sum() >= max(10, len(labels) * 2):
        X, y, signers = X[keep], y[keep], signers[keep]

    train_idx, val_idx, split_method = make_split(y, signers, args.test_size, args.seed)
    train_ds = AugmentedSequenceDataset(
        X[train_idx],
        y[train_idx],
        augment_copies=args.augment_copies,
        noise_std=args.noise_std,
        scale_std=args.scale_std,
        frame_drop_prob=args.frame_drop_prob,
        max_time_shift=args.max_time_shift,
        min_time_scale=args.min_time_scale,
        max_time_scale=args.max_time_scale,
    )
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, input_dim=X.shape[-1], num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics['train_loss']:.4f} "
            f"top1={metrics['top1']:.3f} top3={metrics['top3']:.3f} f1={metrics['macro_f1']:.3f}"
        )
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "input_dim": X.shape[-1],
                    "num_classes": len(labels),
                },
                args.out_dir / "best.pt",
            )

    write_json(args.out_dir / "labels.json", labels)
    write_json(
        args.out_dir / "config.json",
        {
            "model": args.model,
            "input_dim": int(X.shape[-1]),
            "num_classes": len(labels),
            "sequence_length": int(X.shape[1]),
            "schema_version": schema_version,
            "feature_metadata": feature_metadata,
            "confidence_threshold": float(args.confidence_threshold),
            "split_method": split_method,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "effective_train_samples": int(len(train_ds)),
            "augmentation": {
                "augment_copies": int(args.augment_copies),
                "noise_std": float(args.noise_std),
                "scale_std": float(args.scale_std),
                "frame_drop_prob": float(args.frame_drop_prob),
                "max_time_shift": int(args.max_time_shift),
                "min_time_scale": float(args.min_time_scale),
                "max_time_scale": float(args.max_time_scale),
                "label_smoothing": float(args.label_smoothing),
            },
        },
    )
    write_json(args.out_dir / "metrics.json", {"history": history, "best_top1": best_top1})
    print(f"Best validation top-1: {best_top1:.3f}")
    print(f"Artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
