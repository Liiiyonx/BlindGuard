# -*- coding: utf-8 -*-
"""Deterministic unit tests for TrafficLightClassifier (HSV heuristics).

用 numpy 构造纯色小图，验证 red/yellow/green/unknown 四态、
min_pixels 阈值以及多候选时的选择逻辑。不加载模型、不做网络/磁盘 IO。
"""

import unittest

import numpy as np

from core.traffic_light import (
    DEFAULT_COLOR_RANGES,
    STATE_CN,
    TrafficLightClassifier,
)

# 纯色 BGR -> OpenCV HSV：红 H=0、黄 H=30、绿 H=60；灰 S=0 不落入任何区间
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
FULL_BBOX = [0, 0, 10, 10]


def _frame(h=10, w=10, color=GRAY):
    return np.full((h, w, 3), color, dtype=np.uint8)


class SolidColorTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TrafficLightClassifier()

    def test_solid_red_yellow_green_are_classified(self):
        for color, state in ((RED, 'red'), (YELLOW, 'yellow'), (GREEN, 'green')):
            result = self.classifier.classify(_frame(color=color), FULL_BBOX)
            self.assertEqual(result['state'], state)
            self.assertEqual(result['score'], 1.0)

    def test_achromatic_roi_is_unknown(self):
        result = self.classifier.classify(_frame(color=GRAY), FULL_BBOX)
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(result['score'], 0.0)

    def test_dark_red_below_value_threshold_is_unknown(self):
        # (0,0,50): V=50 < red 的 v_min=70，不应判为红
        result = self.classifier.classify(_frame(color=(0, 0, 50)), FULL_BBOX)
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(result['score'], 0.0)


class MinPixelsTests(unittest.TestCase):
    def test_threshold_boundary_between_nine_and_ten_pixels(self):
        frame = _frame(color=GRAY)
        frame[0, :9] = RED
        classifier = TrafficLightClassifier(min_pixels=10)

        below = classifier.classify(frame, FULL_BBOX)
        self.assertEqual(below['state'], 'unknown')
        self.assertEqual(below['score'], 0.0)

        frame[0, 9] = RED  # 恰好 10 像素
        exact = classifier.classify(frame, FULL_BBOX)
        self.assertEqual(exact['state'], 'red')
        self.assertAlmostEqual(exact['score'], 0.1, places=4)

    def test_single_pixel_allowed_with_min_pixels_one(self):
        frame = _frame(h=4, w=4, color=GRAY)
        frame[0, 0] = RED
        result = TrafficLightClassifier(min_pixels=1).classify(
            frame, [0, 0, 4, 4])
        self.assertEqual(result['state'], 'red')
        self.assertEqual(result['score'], round(1 / 16, 4))

    def test_min_pixels_is_clamped_to_at_least_one(self):
        self.assertEqual(TrafficLightClassifier().min_pixels, 10)
        self.assertEqual(TrafficLightClassifier(min_pixels=0).min_pixels, 1)
        self.assertEqual(TrafficLightClassifier(min_pixels=-5).min_pixels, 1)


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TrafficLightClassifier()

    def test_more_pixels_wins(self):
        frame = _frame(color=GRAY)
        frame[0, :4] = RED       # 4 像素
        frame[1, :10] = YELLOW   # 10 像素
        result = self.classifier.classify(frame, FULL_BBOX)
        self.assertEqual(result['state'], 'yellow')
        self.assertAlmostEqual(result['score'], 0.1, places=4)

        frame2 = _frame(color=GRAY)
        frame2[0, :10] = RED
        frame2[1, :4] = YELLOW
        self.assertEqual(self.classifier.classify(frame2, FULL_BBOX)['state'],
                         'red')

    def test_tie_prefers_first_declared_range(self):
        frame = _frame(h=10, w=20, color=GRAY)
        frame[:, :10] = RED      # 100 像素
        frame[:, 10:] = YELLOW   # 100 像素，并列
        result = self.classifier.classify(frame, [0, 0, 20, 10])
        self.assertEqual(result['state'], 'red')


class BboxHandlingTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TrafficLightClassifier()
        self.frame = _frame(color=RED)

    def test_bbox_is_clipped_to_frame(self):
        result = self.classifier.classify(self.frame, [-5, -5, 5, 5])
        # 裁剪为 [0,0,5,5]，整个 ROI 都是红
        self.assertEqual(result['state'], 'red')
        self.assertEqual(result['score'], 1.0)

    def test_bbox_fully_outside_frame_is_unknown(self):
        result = self.classifier.classify(self.frame, [20, 20, 30, 30])
        self.assertEqual(result['state'], 'unknown')

    def test_too_small_bbox_is_unknown(self):
        small = self.classifier.classify(self.frame, [0, 0, 3, 3])
        self.assertEqual(small['state'], 'unknown')
        boundary = self.classifier.classify(self.frame, [0, 0, 4, 4])
        self.assertEqual(boundary['state'], 'red')

    def test_score_uses_roi_area_not_frame_area(self):
        frame = _frame(h=10, w=20, color=GRAY)
        frame[:, :10] = RED
        result = self.classifier.classify(frame, [0, 0, 10, 10])
        self.assertEqual(result['score'], 1.0)

    def test_invalid_inputs_return_unknown(self):
        for frame, bbox in (
            (None, FULL_BBOX),
            (self.frame, None),
            (self.frame, []),
            (self.frame, [0, 0, 10]),
            (self.frame, ['a', 'b', 'c', 'd']),
        ):
            result = self.classifier.classify(frame, bbox)
            self.assertEqual(result['state'], 'unknown')
            self.assertEqual(result['score'], 0.0)


class CustomRangeTests(unittest.TestCase):
    def test_color_ranges_replace_defaults(self):
        classifier = TrafficLightClassifier(color_ranges={'red': [(0, 13, 60, 70)]})
        self.assertEqual(
            classifier.classify(_frame(color=RED), FULL_BBOX)['state'], 'red')
        # 未声明的绿色区间 -> unknown
        self.assertEqual(
            classifier.classify(_frame(color=GREEN), FULL_BBOX)['state'],
            'unknown')

    def test_color_ranges_are_deep_copied(self):
        ranges = {'red': [(0, 13, 60, 70)]}
        classifier = TrafficLightClassifier(color_ranges=ranges)
        ranges['red'].append((177, 180, 60, 70))
        self.assertEqual(len(classifier._ranges['red']), 1)

    def test_default_range_constants_and_state_cn(self):
        self.assertEqual(DEFAULT_COLOR_RANGES['red'][0], (0, 13, 60, 70))
        self.assertEqual(DEFAULT_COLOR_RANGES['yellow'][0], (20, 40, 60, 70))
        self.assertEqual(DEFAULT_COLOR_RANGES['green'][0], (40, 90, 40, 60))
        self.assertEqual(STATE_CN['unknown'], '灯色未知')


if __name__ == '__main__':
    unittest.main()
