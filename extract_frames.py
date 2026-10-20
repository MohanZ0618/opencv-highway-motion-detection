"""Extract one representative frame per video in a folder."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_folder", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("screenshots"))
    parser.add_argument("--position", type=float, default=0.7,
                        help="Relative position in each video, from 0 to 1 (default: 0.7)")
    args = parser.parse_args()
    if not args.video_folder.is_dir():
        parser.error("The video folder does not exist.")
    if not 0 <= args.position <= 1:
        parser.error("--position must be between 0 and 1.")

    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(path for path in args.video_folder.iterdir()
                    if path.is_file() and path.suffix.lower() in extensions)
    if not videos:
        parser.error("No supported video files found in the selected folder.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or total_frames <= 0:
                print(f"Cannot read video metadata: {video.name}")
                failures += 1
                continue
            frame_index = min(int(total_frames * args.position), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(f"Cannot read frame {frame_index}: {video.name}")
                failures += 1
                continue
            output = args.output_dir / f"{video.stem}_frame.jpg"
            if not cv2.imwrite(str(output), frame):
                print(f"Cannot write screenshot: {output}")
                failures += 1
                continue
            print(f"Saved {output} (frame {frame_index})")
        finally:
            cap.release()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
