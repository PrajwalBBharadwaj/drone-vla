"""Actual OpenCV target motion, partial visibility and conservative recovery."""
import unittest

import numpy as np
from PIL import Image

from navigation_perception import FeatureTracker, TrackingLost


WIDTH, HEIGHT = 448, 252
PATCH = np.random.default_rng(570).integers(20, 240, (80, 96, 3), dtype=np.uint8)


def scene(x=176, y=80, copies=()):
    image = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
    for px, py in ((x, y), *copies):
        left, top = max(px, 0), max(py, 0)
        right, bottom = min(px+96, WIDTH), min(py+80, HEIGHT)
        if right > left and bottom > top:
            image[top:bottom, left:right] = PATCH[top-py:bottom-py, left-px:right-px]
    return Image.fromarray(image)


def box(x=176, y=80):
    return x/WIDTH, y/HEIGHT, (x+96)/WIDTH, (y+80)/HEIGHT


class TargetTrackingTests(unittest.TestCase):
    def test_actual_flow_tracks_destination_across_image(self):
        tracker = FeatureTracker()
        tracker.start(scene(), box())
        for offset in (4, 8, 12, 16):
            actual = tracker.update(scene(176+offset, 80))
            self.assertAlmostEqual(actual[0], (176+offset)/WIDTH, delta=0.003)
            self.assertAlmostEqual(actual[1], 80/HEIGHT, delta=0.003)
        self.assertGreater(tracker.confidence, 0.5)

    def test_partially_visible_target_at_left_edge_is_still_tracked(self):
        tracker = FeatureTracker()
        tracker.start(scene(8, 80), box(8, 80))
        actual = tracker.update(scene(-6, 80))
        self.assertEqual(actual[0], 0)
        self.assertAlmostEqual(actual[2], 90/WIDTH, delta=0.004)
        self.assertEqual(tracker.last_status, "partially_visible")

    def test_partially_visible_target_at_right_edge_is_still_tracked(self):
        tracker = FeatureTracker()
        tracker.start(scene(344, 80), box(344, 80))
        actual = tracker.update(scene(358, 80))
        self.assertEqual(actual[2], 1)
        self.assertAlmostEqual(actual[0], 358/WIDTH, delta=0.004)
        self.assertEqual(tracker.last_status, "partially_visible")

    def test_blank_frame_means_unconfirmed_and_keeps_last_measurement(self):
        tracker = FeatureTracker()
        tracker.start(scene(), box())
        last_gray, last_box = tracker.gray.copy(), tracker.box
        with self.assertRaises(TrackingLost) as caught:
            tracker.update(Image.new("RGB", (WIDTH, HEIGHT)))
        self.assertEqual(caught.exception.status, "unconfirmed")
        self.assertNotIn("outside", str(caught.exception))
        self.assertEqual(tracker.box, last_box)
        np.testing.assert_array_equal(tracker.gray, last_gray)

    def test_unique_appearance_can_recover_after_occlusion_and_camera_pan(self):
        tracker = FeatureTracker()
        tracker.start(scene(), box())
        with self.assertRaises(TrackingLost):
            tracker.update(Image.new("RGB", (WIDTH, HEIGHT)))
        actual = tracker.update(scene(300, 60))
        self.assertEqual(tracker.last_status, "recovered")
        np.testing.assert_allclose(actual, box(300, 60), atol=0.003)
        self.assertGreater(tracker.confidence, 0.85)

    def test_recovery_rejects_two_identical_destination_candidates(self):
        tracker = FeatureTracker()
        tracker.start(scene(), box())
        tracker.points = None  # No usable optical flow: appearance must be unique.
        with self.assertRaises(TrackingLost) as caught:
            tracker.update(scene(30, 70, copies=((300, 70),)))
        self.assertEqual(caught.exception.status, "unconfirmed")
        self.assertEqual(tracker.box, box())
        actual = tracker.update(scene(300, 70))
        np.testing.assert_allclose(actual, box(300, 70), atol=0.003)

    def test_unrelated_object_never_replaces_target(self):
        tracker = FeatureTracker()
        tracker.start(scene(), box())
        unrelated = Image.fromarray(np.random.default_rng(987).integers(
            20, 240, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        with self.assertRaises(TrackingLost) as caught:
            tracker.update(unrelated)
        self.assertEqual(caught.exception.status, "unconfirmed")

    def test_recovery_rejects_same_texture_with_a_different_object_colour(self):
        pixels = np.random.default_rng(91).integers(20, 240, (80, 96), dtype=np.uint8)
        initial = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
        initial[80:160, 176:272] = 0
        initial[80:160, 176:272, 2] = pixels  # Grounded object is blue.
        changed = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
        changed[80:160, 176:272] = 0
        changed[80:160, 176:272, 0] = pixels  # Same pattern, different red object.
        tracker = FeatureTracker()
        tracker.start(Image.fromarray(initial), box())
        tracker.points = None
        with self.assertRaises(TrackingLost) as caught:
            tracker.update(Image.fromarray(changed))
        self.assertEqual(caught.exception.status, "unconfirmed")
        self.assertIn("colour", str(caught.exception))

    def test_optical_flow_cannot_accept_a_different_colour_with_same_luminance(self):
        pixels = np.random.default_rng(91).integers(20, 240, (80, 96), dtype=np.uint8)
        initial = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
        initial[80:160, 176:272] = 0
        initial[80:160, 176:272, 2] = pixels
        changed = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
        changed[80:160, 176:272] = 0
        # Match blue's grayscale intensity so flow and gray appearance checks
        # alone would confidently accept a different red destination.
        changed[80:160, 176:272, 0] = np.round(pixels*.114/.299).astype(np.uint8)
        tracker = FeatureTracker()
        tracker.start(Image.fromarray(initial), box())
        with self.assertRaises(TrackingLost) as caught:
            tracker.update(Image.fromarray(changed))
        self.assertEqual(caught.exception.status, "unconfirmed")
        self.assertIn("colour", str(caught.exception))

    def test_only_full_geometric_exit_is_labelled_out_of_view(self):
        tracker = FeatureTracker()
        self.assertEqual(tracker._visible((-0.1, 0.2, 0.2, 0.6)), (0, 0.2, 0.2, 0.6))
        with self.assertRaises(TrackingLost) as caught:
            tracker._visible((-0.3, 0.2, -0.1, 0.6))
        self.assertEqual(caught.exception.status, "out_of_view")


if __name__ == "__main__":
    unittest.main()
