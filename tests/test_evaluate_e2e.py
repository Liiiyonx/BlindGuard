# -*- coding: utf-8 -*-
"""端到端指标匹配与统计的纯函数测试（不加载 Torch）。"""

import unittest

import evaluate


def box(class_name, bbox, light_state=None):
    return {
        'class_name': class_name,
        'bbox': list(bbox),
        'light_state': light_state,
    }


class BoxIoUTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertAlmostEqual(
            evaluate.box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        self.assertEqual(
            evaluate.box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_boxes_do_not_overlap(self):
        self.assertEqual(
            evaluate.box_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_half_overlap_is_one_third(self):
        # 交集 50，并集 150
        self.assertAlmostEqual(
            evaluate.box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3, places=6)


class MatchFrameTests(unittest.TestCase):
    def test_same_box_different_class_is_not_a_match(self):
        matches, missed, false_positives = evaluate.match_frame(
            [box('car', [0, 0, 10, 10])], [box('truck', [0, 0, 10, 10])])

        self.assertEqual(matches, [])
        self.assertEqual(len(missed), 1)
        self.assertEqual(len(false_positives), 1)

    def test_greedy_prefers_higher_iou(self):
        matches, _missed, false_positives = evaluate.match_frame(
            [box('car', [0, 0, 10, 10])],
            [box('car', [0, 0, 6, 6]), box('car', [0, 0, 10, 10])],
            0.3,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1]['bbox'], [0, 0, 10, 10])
        self.assertEqual(len(false_positives), 1)

    def test_one_prediction_cannot_match_two_truths(self):
        matches, missed, false_positives = evaluate.match_frame(
            [box('car', [0, 0, 10, 10]), box('car', [0, 0, 10, 10])],
            [box('car', [0, 0, 10, 10])],
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(len(missed), 1)
        self.assertEqual(false_positives, [])

    def test_threshold_boundary_is_inclusive(self):
        truth = [box('car', [0, 0, 10, 10])]
        predictions = [box('car', [5, 0, 15, 10])]  # IoU = 1/3

        self.assertEqual(
            len(evaluate.match_frame(truth, predictions, 0.34)[0]), 0)
        self.assertEqual(
            len(evaluate.match_frame(truth, predictions, 1 / 3)[0]), 1)

    def test_invalid_boxes_are_dropped(self):
        self.assertEqual(
            evaluate._normalize_boxes(
                [{'class_name': 'car', 'bbox': [1, 2]}]), [])
        self.assertEqual(
            evaluate._normalize_boxes([{'class_name': 'car'}]), [])
        self.assertEqual(evaluate._normalize_boxes(None), [])

    def test_normalized_boxes_keep_light_state(self):
        normalized = evaluate._normalize_boxes([
            box('traffic_light', [1, 2, 3, 4], 'red'),
        ])

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]['light_state'], 'red')
        self.assertEqual(normalized[0]['bbox'], [1.0, 2.0, 3.0, 4.0])


class PrfTests(unittest.TestCase):
    def test_no_samples_returns_zeros(self):
        self.assertEqual(evaluate.prf_counts(0, 0, 0), (0.0, 0.0, 0.0))

    def test_perfect_match(self):
        self.assertEqual(evaluate.prf_counts(5, 0, 0), (1.0, 1.0, 1.0))

    def test_all_missed(self):
        self.assertEqual(evaluate.prf_counts(0, 0, 4), (0.0, 0.0, 0.0))

    def test_known_values(self):
        precision, recall, _ = evaluate.prf_counts(3, 1, 2)
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.6)


if __name__ == '__main__':
    unittest.main()
