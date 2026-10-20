"""Create an original synthetic video with one moving and one stationary shape."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo.mp4"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (640, 360))
    if not writer.isOpened():
        raise SystemExit("Cannot create the demo video. Check mp4v codec support.")
    try:
        for index in range(120):
            frame = np.full((360, 640, 3), 255, dtype=np.uint8)
            cv2.circle(frame, (550, 275), 35, (0, 0, 0), thickness=-1)
            x = 30 + (index * 16) % 440
            cv2.rectangle(frame, (x, 100), (x + 80, 160), (170, 95, 25), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Saved {args.output}: 120 frames at 30 FPS, 640 x 360")


if __name__ == "__main__":
    main()
