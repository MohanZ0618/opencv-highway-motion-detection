import cv2
import numpy as np
import os
import argparse

# =========================================================
# (2) MOG2 Combined with Frame Differencing
# Pixel-level intersection
# =========================================================

parser = argparse.ArgumentParser(description='MOG2 and frame-difference intersection baseline')
parser.add_argument("video", help="Path to the input video")
parser.add_argument("--output-dir", default="outputs", help="Folder for exported videos")
parser.add_argument("--no-save", action="store_true", help="Disable video export")
parser.add_argument("--no-display", action="store_true", help="Process without GUI windows")
args = parser.parse_args()
video_path = args.video

if not os.path.exists(video_path):
    print("Video file does not exist:", video_path)
    raise SystemExit(1)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file:", video_path)
    raise SystemExit(1)

ret, prev_frame = cap.read()
if not ret:
    print("Error reading first frame.")
    cap.release()
    raise SystemExit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0)

# MOG2
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

# Parameters
bg_threshold = 180
diff_threshold = 20

min_contour_area = 500
min_width = 15
min_height = 15

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

save_output = not args.no_save
out = None

if save_output:
    os.makedirs(args.output_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    output_path = os.path.join(args.output_dir, "mog2_plus_frame_difference_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if save_output:
    writers = [out]
    if not all(writer.isOpened() for writer in writers):
        cap.release()
        for writer in writers:
            writer.release()
        raise SystemExit("Cannot initialize video output. Check the output folder and mp4v codec support.")

warmup_frames = 20
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Frame differencing branch
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel_open)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel_close)
    diff_mask = diff_mask.astype(np.uint8)

    # MOG2 branch
    fg_mask = backSub.apply(frame)
    _, bg_mask = cv2.threshold(fg_mask, bg_threshold, 255, cv2.THRESH_BINARY)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_open)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close)
    bg_mask = bg_mask.astype(np.uint8)

    if frame_count <= warmup_frames:
        if not args.no_display:
            cv2.imshow("MOG2 Mask", bg_mask)
            cv2.imshow("Frame Difference Mask", diff_mask)
            cv2.imshow("Final Intersection Mask", np.zeros_like(bg_mask))
            cv2.imshow("MOG2 + Frame Difference Output", frame)

        prev_gray = gray.copy()

        if not args.no_display and cv2.waitKey(30) & 0xFF == 27:
            break
        continue

    # Pixel-level intersection
    final_mask = cv2.bitwise_and(bg_mask, diff_mask)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    final_mask = final_mask.astype(np.uint8)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_frame = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)

            if w > min_width and h > min_height:
                aspect_ratio = w / float(h)
                if 0.6 < aspect_ratio < 4.0:
                    cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if not args.no_display:
        cv2.imshow("MOG2 Mask", bg_mask)
        cv2.imshow("Frame Difference Mask", diff_mask)
        cv2.imshow("Final Intersection Mask", final_mask)
        cv2.imshow("MOG2 + Frame Difference Output", output_frame)

    if save_output and out is not None:
        out.write(output_frame)

    prev_gray = gray.copy()

    if not args.no_display and cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
if out is not None:
    out.release()
if not args.no_display:
    cv2.destroyAllWindows()
