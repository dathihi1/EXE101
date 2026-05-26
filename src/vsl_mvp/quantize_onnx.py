from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamically quantize an ONNX classifier to INT8 where supported.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(args.model), str(args.out), weight_type=QuantType.QInt8)
    before = args.model.stat().st_size / (1024 * 1024)
    after = args.out.stat().st_size / (1024 * 1024)
    print(f"Quantized {args.model} -> {args.out}")
    print(f"Size: {before:.2f} MB -> {after:.2f} MB")


if __name__ == "__main__":
    main()
