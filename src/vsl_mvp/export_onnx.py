from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models import build_model
from .utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained PyTorch checkpoint to ONNX.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = read_json(args.run_dir / "config.json")
    try:
        checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    model = build_model(config["model"], config["input_dim"], config["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.zeros(1, config["sequence_length"], config["input_dim"], dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {args.out}")


if __name__ == "__main__":
    main()
