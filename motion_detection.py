import cv2
import numpy as np
import os
import argparse

# =========================================================
# Motion Detection using MOG2 Only
# 1. Background subtraction with MOG2
# 2. Thresholding
# 3. Morphological filtering
# 4. Build bg_contour_mask
# 5. Remove thin railing-like connected structures
# 6. Use lightly processed bg mask to recover missed small cars
# 7. Delay blue-box branch to avoid early railing detections
# =========================================================

parser = argparse.ArgumentParser(description='MOG2 highway motion detection')
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

# MOG2 background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

# -----------------------------
# Parameters
# -----------------------------
bg_threshold = 180

min_contour_area = 1100
min_width = 30
min_height = 30

# lighter rules for recovering small cars from bg_light_mask
small_contour_area = 500
small_min_width = 18
small_min_height = 18

aspect_ratio_min = 0.6
aspect_ratio_max = 3.5
extent_threshold = 0.2
small_extent_threshold = 0.12

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# used to remove thin connected lines
kernel_remove_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# light kernel for bg branch
kernel_light = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

warmup_frames = 20
light_branch_start_frame = 50

# Optional output video
save_output = not args.no_save
out_final = None
out_fg = None
out_bg = None
out_bg_contour = None
out_refined = None
out_light = None

if save_output:
    os.makedirs(args.output_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_final = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_final.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    out_fg = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_fg_mask.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    out_bg = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_bg_mask.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    out_bg_contour = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_bg_contour_mask.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    out_refined = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_refined_bg_contour_mask.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    out_light = cv2.VideoWriter(
        os.path.join(args.output_dir, "output_bg_light_mask.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )

if save_output:
    writers = [out_final, out_fg, out_bg, out_bg_contour, out_refined, out_light]
    if not all(writer.isOpened() for writer in writers):
        cap.release()
        for writer in writers:
            writer.release()
        raise SystemExit("Cannot initialize video output. Check the output folder and mp4v codec support.")

frame_count = 0


def get_boxes_from_mask(mask, min_area, min_w, min_h, extent_min):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)

            if w > min_w and h > min_h:
                aspect_ratio = w / float(h)
                extent = area / float(w * h)

                if aspect_ratio_min < aspect_ratio < aspect_ratio_max and extent > extent_min:
                    boxes.append((x, y, w, h))
    return boxes


def boxes_overlap(boxA, boxB, overlap_thresh=0.3):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    if areaA == 0 or areaB == 0:
        return False

    overlap_ratio = inter_area / float(min(areaA, areaB))
    return overlap_ratio > overlap_thresh

def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# main ,reading frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # -----------------------------
    # Background subtraction branch
    # -----------------------------
    fg_mask = backSub.apply(frame, learningRate=0.02)
    _, bg_mask = cv2.threshold(fg_mask, bg_threshold, 255, cv2.THRESH_BINARY)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_open)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close)
    bg_mask = bg_mask.astype(np.uint8)

    if frame_count <= warmup_frames:
        empty_mask = np.zeros_like(bg_mask)

        if not args.no_display:
            cv2.imshow("FG Mask", fg_mask)
            cv2.imshow("BG Mask", bg_mask)
            cv2.imshow("BG Light Mask", empty_mask)
            cv2.imshow("BG Contour Mask", empty_mask)
            cv2.imshow("Refined BG Contour Mask", empty_mask)
            cv2.imshow("Motion Detection Output", frame)

        if save_output:
            if out_fg is not None:
                out_fg.write(mask_to_bgr(fg_mask))
            if out_bg is not None:
                out_bg.write(mask_to_bgr(bg_mask))
            if out_light is not None:
                out_light.write(mask_to_bgr(empty_mask))
            if out_bg_contour is not None:
                out_bg_contour.write(mask_to_bgr(empty_mask))
            if out_refined is not None:
                out_refined.write(mask_to_bgr(empty_mask))
            if out_final is not None:
                out_final.write(frame)

        if not args.no_display and cv2.waitKey(30) & 0xFF == 27:
            break
        continue
    # -----------------------------
    # Build bg_contour_mask
    # -----------------------------
    bg_contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bg_contour_mask = np.zeros_like(bg_mask)

    for cnt in bg_contours:
        area = cv2.contourArea(cnt)

        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)

            if w > min_width and h > min_height:
                aspect_ratio = w / float(h)
                extent = area / float(w * h)

                if aspect_ratio_min < aspect_ratio < aspect_ratio_max and extent > extent_threshold:
                    cv2.drawContours(bg_contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # -----------------------------
    # Refined branch
    # -----------------------------
    refined_bg_contour_mask = cv2.morphologyEx(
        bg_contour_mask,
        cv2.MORPH_OPEN,
        kernel_remove_line
    )

    refined_bg_contour_mask = cv2.morphologyEx(
        refined_bg_contour_mask,
        cv2.MORPH_CLOSE,
        kernel_open
    )

    refined_bg_contour_mask = refined_bg_contour_mask.astype(np.uint8)

    # -----------------------------
    # Light bg branch for missed small cars
    # -----------------------------
    bg_light_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_light)
    bg_light_mask = bg_light_mask.astype(np.uint8)

    # -----------------------------
    # Get boxes
    # -----------------------------
    refined_boxes = get_boxes_from_mask(
        refined_bg_contour_mask,
        min_contour_area,
        min_width,
        min_height,
        extent_threshold
    )

    light_boxes = get_boxes_from_mask(
        bg_light_mask,
        small_contour_area,
        small_min_width,
        small_min_height,
        small_extent_threshold
    )

    output_frame = frame.copy()

    # -----------------------------
    # Draw refined boxes first (red)
    # -----------------------------
    for box in refined_boxes:
        x, y, w, h = box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # -----------------------------
    # Draw blue boxes only after delayed start
    # -----------------------------
    if frame_count >= light_branch_start_frame:
        for lbox in light_boxes:
            overlap_found = False
            for rbox in refined_boxes:
                if boxes_overlap(lbox, rbox, overlap_thresh=0.3):
                    overlap_found = True
                    break

            if not overlap_found:
                x, y, w, h = lbox
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if not args.no_display:
        cv2.imshow("FG Mask", fg_mask)
        cv2.imshow("BG Mask", bg_mask)
        cv2.imshow("BG Light Mask", bg_light_mask)
        cv2.imshow("BG Contour Mask", bg_contour_mask)
        cv2.imshow("Refined BG Contour Mask", refined_bg_contour_mask)
        cv2.imshow("Motion Detection Output", output_frame)

    if save_output:
        if out_final is not None:
            out_final.write(output_frame)
        if out_fg is not None:
            out_fg.write(mask_to_bgr(fg_mask))
        if out_bg is not None:
            out_bg.write(mask_to_bgr(bg_mask))
        if out_bg_contour is not None:
            out_bg_contour.write(mask_to_bgr(bg_contour_mask))
        if out_refined is not None:
            out_refined.write(mask_to_bgr(refined_bg_contour_mask))
        if out_light is not None:
            out_light.write(mask_to_bgr(bg_light_mask))

    if not args.no_display and cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()

if out_final is not None:
    out_final.release()
if out_fg is not None:
    out_fg.release()
if out_bg is not None:
    out_bg.release()
if out_bg_contour is not None:
    out_bg_contour.release()
if out_refined is not None:
    out_refined.release()
if out_light is not None:
    out_light.release()

if not args.no_display:
    cv2.destroyAllWindows()
