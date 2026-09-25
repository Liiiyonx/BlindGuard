# -*- coding: utf-8 -*-
"""Deterministic unit tests for SimpleTracker (IoU matching + motion trend).

不加载模型、不访问网络、不写磁盘；时间通过替换 core.tracker.time 完全可控，
因此无需 sleep 即可稳定复现趋势窗口与轨迹过期行为。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.geometry import box_iou
from core.tracker import SimpleTracker, TREND_CN


class _Clock:
    """可手动推进的 time.time 替身（确定性，不依赖真实时间）。"""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def _det(class_name='car', bbox=(0, 0, 10, 10), area=None, confidence=0.5):
    det = {'class_name': class_name, 'bbox': list(bbox),
           'confidence': confidence}
    if area is not None:
        det['area'] = area
    return det


class IouTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap_value(self):
        # inter = 5*10 = 50, union = 100 + 100 - 50 = 150 -> 1/3
        self.assertAlmostEqual(
            box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150, places=6)

    def test_degenerate_box_has_zerobox_iou(self):
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 10, 10]), 0.0)


class TrackerTestCase(unittest.TestCase):
    """公共夹具：用可推进时钟替换 core.tracker.time。"""

    def setUp(self):
        self.clock = _Clock()
        patcher = patch('core.tracker.time',
                        SimpleNamespace(time=self.clock))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tracker = SimpleTracker()


class AnnotationTests(TrackerTestCase):
    def test_new_detection_creates_track_and_annotates_fields(self):
        det = _det(confidence=0.5)
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 1)
        self.assertEqual(det['track_age'], 0.0)
        self.assertEqual(det['trend'], 'stable')
        self.assertEqual(det['conf_stable'], 0.5)
        self.assertEqual(det['conf_smooth'], 0.5)
        self.assertEqual(self.tracker.active_count, 1)

    def test_same_object_across_frames_keeps_id(self):
        self.tracker.update([_det(bbox=(0, 0, 10, 10))])
        self.clock.advance(0.05)

        det = _det(bbox=(1, 1, 11, 11))  # IoU ~0.68 > 0.3
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 1)
        self.assertEqual(self.tracker.active_count, 1)

    def test_different_class_never_matches_existing_track(self):
        self.tracker.update([_det('car')])
        self.clock.advance(0.05)

        det = _det('person')  # 相同框、不同类别
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 2)
        self.assertEqual(self.tracker.active_count, 2)

    def test_non_overlapping_same_class_creates_new_track(self):
        self.tracker.update([_det('car', bbox=(0, 0, 10, 10))])
        self.clock.advance(0.05)

        det = _det('car', bbox=(100, 100, 110, 110))  # IoU == 0
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 2)

    def test_iou_at_or_below_threshold_is_not_matched(self):
        # [5,0,15,10] -> IoU=1/3 (>0.3) 匹配；[8,0,18,10] -> IoU=1/9 (<0.3) 不匹配
        self.tracker.update([_det('car', bbox=(0, 0, 10, 10))])
        self.clock.advance(0.05)
        matched = _det('car', bbox=(5, 0, 15, 10))
        self.tracker.update([matched])
        self.assertEqual(matched['track_id'], 1)

        other = SimpleTracker()
        other.update([_det('car', bbox=(0, 0, 10, 10))])
        self.clock.advance(0.05)
        unmatched = _det('car', bbox=(8, 0, 18, 10))
        other.update([unmatched])
        self.assertEqual(unmatched['track_id'], 2)

    def test_greedy_matching_assigns_each_track_once(self):
        self.tracker.update([_det('car', bbox=(0, 0, 10, 10)),
                             _det('car', bbox=(20, 0, 30, 10))])
        self.clock.advance(0.05)

        d1 = _det('car', bbox=(1, 1, 11, 11))
        d2 = _det('car', bbox=(21, 1, 31, 11))
        self.tracker.update([d1, d2])

        self.assertEqual({d1['track_id'], d2['track_id']}, {1, 2})

    def test_conf_smooth_is_historical_max_not_a_smoothed_value(self):
        self.tracker.update([_det(confidence=0.4)])
        self.clock.advance(0.05)
        self.tracker.update([_det(confidence=0.9)])
        self.clock.advance(0.05)

        det = _det(confidence=0.2)
        self.tracker.update([det])

        # 历史最高分被保留，当前帧的低分不会拉低 conf_stable / conf_smooth
        self.assertEqual(det['conf_stable'], 0.9)
        self.assertEqual(det['conf_smooth'], det['conf_stable'])

    def test_area_falls_back_to_bbox_when_missing_or_zero(self):
        det = _det(bbox=(0, 0, 10, 20))  # 无 area 字段
        self.tracker.update([det])
        self.assertEqual(self.tracker._tracks[1]['history'][-1][1], 200.0)

        zero = SimpleTracker()
        det_zero = _det(bbox=(0, 0, 10, 20), area=0)  # 0 视为缺失
        zero.update([det_zero])
        self.assertEqual(zero._tracks[1]['history'][-1][1], 200.0)

    def test_explicit_area_overrides_bbox_estimate(self):
        det = _det(bbox=(0, 0, 10, 10), area=7.5)
        self.tracker.update([det])
        self.assertEqual(self.tracker._tracks[1]['history'][-1][1], 7.5)

    def test_track_age_accumulates_from_first_seen(self):
        self.tracker.update([_det()])
        self.clock.advance(3.0)
        det = _det(bbox=(1, 1, 11, 11))
        self.tracker.update([det])
        self.assertEqual(det['track_age'], 3.0)

    def test_malformed_detection_does_not_crash(self):
        det = {}
        self.tracker.update([det])
        self.assertEqual(det['track_id'], 1)
        self.assertEqual(det['track_age'], 0.0)
        self.assertEqual(det['trend'], 'stable')

        short = {'class_name': 'car', 'bbox': [1, 2]}
        self.clock.advance(0.05)
        self.tracker.update([short])
        self.assertIn('track_id', short)

    def test_reset_clears_tracks(self):
        self.tracker.update([_det()])
        self.tracker.reset()
        self.assertEqual(self.tracker.active_count, 0)

    def test_trend_cn_mapping(self):
        self.assertEqual(TREND_CN['approaching'], '正在接近')
        self.assertEqual(TREND_CN['receding'], '正在远离')
        self.assertEqual(TREND_CN['stable'], '')


class ExpiryTests(TrackerTestCase):
    def test_unmatched_track_expires_after_max_missing(self):
        self.tracker.update([_det('car', bbox=(0, 0, 10, 10))])
        self.assertEqual(self.tracker.active_count, 1)

        self.clock.advance(2.0)  # > max_missing(1.0)
        det = _det('car', bbox=(500, 500, 510, 510))
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 2)
        self.assertEqual(self.tracker.active_count, 1)
        self.assertNotIn(1, self.tracker._tracks)

    def test_matched_track_is_not_expired(self):
        self.tracker.update([_det('car', bbox=(0, 0, 10, 10))])
        self.clock.advance(2.0)

        det = _det('car', bbox=(1, 1, 11, 11))  # 同一目标，重新命中
        self.tracker.update([det])

        self.assertEqual(det['track_id'], 1)
        self.assertEqual(self.tracker.active_count, 1)


class TrendTests(TrackerTestCase):
    def setUp(self):
        super().setUp()
        self.tracker = SimpleTracker(trend_window=1.0, trend_ratio=1.25)

    def _two_frame_trend(self, area_before, area_after, advance=1.5):
        self.tracker.update([_det(area=area_before)])
        self.clock.advance(advance)
        det = _det(area=area_after)
        self.tracker.update([det])
        return det['trend']

    def test_approaching_when_area_grows(self):
        self.assertEqual(self._two_frame_trend(100.0, 130.0), 'approaching')

    def test_receding_when_area_shrinks(self):
        self.assertEqual(self._two_frame_trend(100.0, 70.0), 'receding')

    def test_stable_within_ratio_band(self):
        self.assertEqual(self._two_frame_trend(100.0, 110.0), 'stable')

    def test_trend_boundary_ratios_are_inclusive(self):
        # ratio >= 1.25 -> approaching；ratio <= 1/1.25 -> receding
        self.assertEqual(self._two_frame_trend(100.0, 125.0), 'approaching')
        self.tracker = SimpleTracker(trend_window=1.0, trend_ratio=1.25)
        self.assertEqual(self._two_frame_trend(100.0, 80.0), 'receding')

    def test_stable_when_no_sample_older_than_window(self):
        # 第二帧距首帧仅 0.5s (< trend_window)，没有可用的历史参照
        self.assertEqual(
            self._two_frame_trend(100.0, 200.0, advance=0.5), 'stable')


if __name__ == '__main__':
    unittest.main()
