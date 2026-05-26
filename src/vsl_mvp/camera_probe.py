from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local cameras and optionally save a test frame.")
    parser.add_argument("--max-index", default=5, type=int)
    parser.add_argument("--save", default="camera_probe.jpg", type=Path)
    args = parser.parse_args()

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    saved = False
    for idx in range(args.max_index):
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        ok = False
        shape = None
        if opened:
            time.sleep(0.5)
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = frame.shape
                if not saved:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.save), frame)
                    saved = True
        cap.release()
        print(f"camera {idx}: opened={opened} read={ok} shape={shape}")
    if saved:
        print(f"Saved test frame: {args.save.resolve()}")


if __name__ == "__main__":
    main()
