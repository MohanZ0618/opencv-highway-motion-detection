import cv2
import numpy as np
import os
import argparse

# =========================================================
# (1) Pure Frame Differencing
# =========================================================

parser = argparse.ArgumentParser(description='Pure frame-differencing baseline')
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

# Parameters
diff_threshold = 20
min_contour_area = 700
min_width = 20
min_height = 20

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Output video
save_output = not args.no_save
out = None

if save_output:
    os.makedirs(args.output_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Detected input FPS =", fps)

    if fps <= 1 or fps > 120:
        fps = 10
        print("Using fallback FPS =", fps)

    output_path = os.path.join(args.output_dir, "pure_frame_differencing_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out.write(prev_frame)

if save_output:
    writers = [out]
    if not all(writer.isOpened() for writer in writers):
        cap.release()
        for writer in writers:
            writer.release()
        raise SystemExit("Cannot initialize video output. Check the output folder and mp4v codec support.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Frame differencing
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Thresholding
    _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # Morphology
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel_open)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel_close)
    diff_mask = diff_mask.astype(np.uint8)

    # Contours
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        cv2.imshow("Frame Difference Mask", diff_mask)
        cv2.imshow("Pure Frame Differencing Output", output_frame)

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
