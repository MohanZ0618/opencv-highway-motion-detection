"""Exercise the public commands using an original, generated video."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def run_command(self, *arguments, expected_code=0):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        return result

    def read_video(self, path):
        cap = cv2.VideoCapture(str(path))
        self.assertTrue(cap.isOpened(), str(path))
        count = 0
        has_red_box = False
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.assertEqual(frame.shape[:2], (360, 640))
                red = frame[:, :, 2].astype(np.int16)
                green = frame[:, :, 1].astype(np.int16)
                blue = frame[:, :, 0].astype(np.int16)
                red_pixels = np.count_nonzero((red > green + 60) & (red > blue + 60))
                if count < 20:
                    self.assertEqual(red_pixels, 0)
                else:
                    has_red_box |= red_pixels > 50
                count += 1
        finally:
            cap.release()
        return count, has_red_box

    def test_end_to_end_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "demo.mp4"
            self.run_command(ROOT / "examples/generate_demo.py", "--output", source)
            output = directory / "main"
            self.run_command(ROOT / "motion_detection.py", source,
                             "--output-dir", output, "--no-display")
            expected = {"output_final.mp4", "output_fg_mask.mp4", "output_bg_mask.mp4",
                        "output_bg_contour_mask.mp4", "output_refined_bg_contour_mask.mp4",
                        "output_bg_light_mask.mp4"}
            self.assertEqual({p.name for p in output.glob("*.mp4")}, expected)
            for name in expected:
                count, has_red_box = self.read_video(output / name)
                self.assertEqual(count, 120, name)
                if name == "output_final.mp4":
                    self.assertTrue(has_red_box, "No main-branch detections on the moving shape")

            for script, filename, frame_count in [
                ("frame_difference.py", "pure_frame_differencing_output.mp4", 120),
                ("mog2_frame_difference.py", "mog2_plus_frame_difference_output.mp4", 99),
            ]:
                baseline_output = directory / script.removesuffix(".py")
                self.run_command(ROOT / script, source, "--output-dir", baseline_output,
                                 "--no-display")
                cap = cv2.VideoCapture(str(baseline_output / filename))
                try:
                    self.assertTrue(cap.isOpened())
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_count)
                finally:
                    cap.release()

            screenshots = directory / "screenshots"
            self.run_command(ROOT / "extract_frames.py", directory,
                             "--output-dir", screenshots, "--position", "0.5")
            self.assertIsNotNone(cv2.imread(str(screenshots / "demo_frame.jpg")))

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.mp4"
            for script in ("motion_detection.py", "frame_difference.py", "mog2_frame_difference.py"):
                result = self.run_command(ROOT / script, missing, "--no-display", expected_code=1)
                self.assertIn("does not exist", result.stdout)
            self.run_command(ROOT / "extract_frames.py", temp, "--position", "1.1", expected_code=2)


if __name__ == "__main__":
    unittest.main()
