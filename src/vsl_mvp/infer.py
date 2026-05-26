from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import read_json


class OnnxSignRecognizer:
    def __init__(self, model_path: str | Path, labels_path: str | Path, config_path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.labels = read_json(labels_path)
        self.config = read_json(config_path)
        self.threshold = float(self.config.get("confidence_threshold", 0.55))
        self.margin_threshold = float(self.config.get("confidence_margin_threshold", 0.0))

    def predict(self, sequence: np.ndarray) -> dict:
        x = sequence.astype(np.float32)[None, ...]
        logits = self.session.run(None, {"sequence": x})[0][0]
        probs = softmax(logits)
        order = np.argsort(probs)[::-1]
        top3 = [{"label": self.labels[int(i)], "confidence": float(probs[int(i)])} for i in order[:3]]
        best = top3[0]
        margin = best["confidence"] - (top3[1]["confidence"] if len(top3) > 1 else 0.0)
        if best["confidence"] < self.threshold:
            status = "low_confidence"
        elif margin < self.margin_threshold:
            status = "low_margin"
        else:
            status = "ok"
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "confidence_margin": float(margin),
            "top3": top3,
            "status": status,
        }


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - np.max(logits)
    exp = np.exp(values)
    return exp / exp.sum()
