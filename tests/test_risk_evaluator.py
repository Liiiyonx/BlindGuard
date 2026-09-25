# -*- coding: utf-8 -*-
"""Deterministic unit tests for RiskEvaluator.

覆盖类别风险优先级、距离/位置分档、权重合成、红灯加权与风险等级分桶。
纯内存计算，无模型/网络/磁盘依赖。
"""

import unittest

from core.risk_evaluator import RiskEvaluator, RiskLevel


def _det(class_name='car', area_ratio=0.0, center=(0, 0), **extra):
    det = {'class_name': class_name, 'area_ratio': area_ratio,
           'center': center}
    det.update(extra)
    return det


class ClassRiskTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_high_risk_set_scores_0_90(self):
        for name in ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train',
                     'fire hydrant'):
            self.assertEqual(
                self.evaluator._calculate_class_risk(name), 0.90, name)

    def test_medium_risk_set_scores_0_60(self):
        for name in ('person', 'dog', 'cat', 'chair', 'bench', 'potted plant',
                     'suitcase', 'backpack', 'umbrella', 'stop sign'):
            self.assertEqual(
                self.evaluator._calculate_class_risk(name), 0.60, name)

    def test_low_risk_set_scores_0_30(self):
        for name in ('bottle', 'cup', 'book', 'cell phone', 'laptop', 'mouse',
                     'keyboard', 'tv', 'remote', 'clock'):
            self.assertEqual(
                self.evaluator._calculate_class_risk(name), 0.30, name)

    def test_explicit_score_table_takes_priority(self):
        # 显式表 > 三个集合；'traffic light' / 'pole' / 'handbag' / 'bird' 均只登记在表中
        self.assertEqual(self.evaluator._calculate_class_risk('traffic light'),
                         0.50)
        self.assertEqual(self.evaluator._calculate_class_risk('pole'), 0.40)
        self.assertEqual(self.evaluator._calculate_class_risk('handbag'), 0.15)
        self.assertEqual(self.evaluator._calculate_class_risk('bird'), 0.20)

    def test_unregistered_class_uses_default(self):
        self.assertEqual(
            self.evaluator._calculate_class_risk('manhole_cover'), 0.30)
        self.assertEqual(self.evaluator._calculate_class_risk('default'), 0.30)

    def test_underscore_traffic_light_misses_explicit_table(self):
        # 自定义主模型给出的类别是下划线写法 'traffic_light'，
        # 显式表登记的是带空格的 'traffic light'，因此下划线版本落回 default。
        self.assertEqual(
            self.evaluator._calculate_class_risk('traffic_light'), 0.30)

    def test_class_scores_override_wins_over_sets(self):
        overridden = RiskEvaluator(
            risk_thresholds={'class_scores': {'car': 0.05}})
        self.assertEqual(overridden._calculate_class_risk('car'), 0.05)

    def test_class_scores_can_add_new_entries_and_override_default(self):
        added = RiskEvaluator(
            risk_thresholds={'class_scores': {'manhole_cover': 0.77}})
        self.assertEqual(added._calculate_class_risk('manhole_cover'), 0.77)

        default_overridden = RiskEvaluator(
            risk_thresholds={'class_scores': {'default': 0.99}})
        self.assertEqual(
            default_overridden._calculate_class_risk('manhole_cover'), 0.99)

    def test_custom_sets_replace_defaults(self):
        custom = RiskEvaluator(high_risk_classes=['rock'])
        self.assertEqual(custom._calculate_class_risk('rock'), 0.90)
        # car 不再属于默认高风险集合 -> 落回 default
        self.assertEqual(custom._calculate_class_risk('car'), 0.30)

    def test_threshold_overrides_do_not_pollute_class_or_other_instances(self):
        RiskEvaluator(risk_thresholds={'class_scores': {'car': 0.05}})
        self.assertNotIn('car', RiskEvaluator.CLASS_RISK_SCORES)
        self.assertEqual(RiskEvaluator()._calculate_class_risk('car'), 0.90)

    def test_missing_class_name_falls_back_to_default(self):
        result = self.evaluator._evaluate_single({}, frame_width=640)
        self.assertEqual(result['class_name'], 'default')
        self.assertEqual(result['risk_factors']['class_score'], 0.30)


class DistanceRiskTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_distance_risk_boundaries(self):
        cases = {
            0.25: 1.0, 0.20: 1.0, 0.19: 0.8, 0.10: 0.8, 0.09: 0.5,
            0.04: 0.5, 0.03: 0.2, 0.01: 0.2, 0.005: 0.05, 0.0: 0.05,
        }
        for area_ratio, expected in cases.items():
            self.assertEqual(
                self.evaluator._calculate_distance_risk(area_ratio),
                expected, f'area_ratio={area_ratio}')

    def test_distance_level_boundaries(self):
        cases = {
            0.25: 'very_close', 0.20: 'very_close', 0.19: 'close',
            0.10: 'close', 0.09: 'medium', 0.04: 'medium', 0.03: 'far',
            0.01: 'far', 0.0: 'very_far',
        }
        for area_ratio, expected in cases.items():
            self.assertEqual(
                self.evaluator._determine_distance_level(area_ratio),
                expected, f'area_ratio={area_ratio}')

    def test_distance_override_does_not_pollute_class_attribute(self):
        custom = RiskEvaluator(
            risk_thresholds={'distance': {'very_close': 0.5}})
        self.assertEqual(custom._calculate_distance_risk(0.5), 1.0)
        self.assertEqual(custom._calculate_distance_risk(0.49), 0.8)
        self.assertEqual(
            RiskEvaluator.DISTANCE_THRESHOLDS['very_close'], 0.20)
        self.assertEqual(RiskEvaluator()._calculate_distance_risk(0.5), 1.0)


class PositionRiskTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_position_risk_peaks_at_center(self):
        self.assertEqual(
            self.evaluator._calculate_position_risk(320, 640), 1.0)
        self.assertAlmostEqual(
            self.evaluator._calculate_position_risk(160, 640), 0.75)
        self.assertAlmostEqual(
            self.evaluator._calculate_position_risk(480, 640), 0.75)

    def test_position_risk_is_low_on_sides(self):
        self.assertEqual(
            self.evaluator._calculate_position_risk(64, 640), 0.3)
        self.assertEqual(
            self.evaluator._calculate_position_risk(600, 640), 0.3)

    def test_position_zone_boundaries(self):
        # 中心区 [0.25, 0.75] 闭区间
        self.assertEqual(
            self.evaluator._determine_position_zone(160, 640), 'center')
        self.assertEqual(
            self.evaluator._determine_position_zone(159, 640), 'left')
        self.assertEqual(
            self.evaluator._determine_position_zone(480, 640), 'center')
        self.assertEqual(
            self.evaluator._determine_position_zone(481, 640), 'right')


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_weight_composition_and_level(self):
        # car 0.90*0.45 + distance 1.0*0.35 + position 1.0*0.20 = 0.955
        det = _det('car', area_ratio=0.25, center=(320, 200))
        result = self.evaluator._evaluate_single(det, frame_width=640)

        self.assertEqual(result['risk_factors']['class_score'], 0.90)
        self.assertEqual(result['risk_factors']['distance_score'], 1.0)
        self.assertEqual(result['risk_factors']['position_score'], 1.0)
        self.assertEqual(result['risk_score'], 0.955)
        self.assertEqual(result['risk_level'], 'critical')
        self.assertEqual(result['risk_level_cn'], '极高风险')

    def test_red_light_adds_weight_when_under_cap(self):
        # manhole_cover 0.30*0.45 + 0.5*0.35 + 1.0*0.20 = 0.51 -> medium
        base = _det('manhole_cover', area_ratio=0.04, center=(320, 0))
        plain = self.evaluator._evaluate_single(base, frame_width=640)
        self.assertEqual(plain['risk_score'], 0.51)
        self.assertEqual(plain['risk_level'], 'medium')

        red = _det('manhole_cover', area_ratio=0.04, center=(320, 0),
                   light_state='red')
        weighted = self.evaluator._evaluate_single(red, frame_width=640)
        self.assertEqual(weighted['risk_score'], 0.66)  # 0.51 + 0.15
        self.assertEqual(weighted['risk_level'], 'high')
        self.assertEqual(weighted['light_state'], 'red')

    def test_red_light_weight_is_capped_at_one(self):
        det = _det('car', area_ratio=0.25, center=(320, 0), light_state='red')
        result = self.evaluator._evaluate_single(det, frame_width=640)
        self.assertEqual(result['risk_score'], 1.0)
        self.assertEqual(result['risk_level'], 'critical')

    def test_non_red_light_state_does_not_add_weight(self):
        base = _det('manhole_cover', area_ratio=0.04, center=(320, 0))
        green = dict(base, light_state='green')
        self.assertEqual(
            self.evaluator._evaluate_single(green, frame_width=640)['risk_score'],
            self.evaluator._evaluate_single(base, frame_width=640)['risk_score'])


class RiskLevelBucketTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_level_buckets(self):
        cases = {
            1.0: RiskLevel.CRITICAL, 0.80: RiskLevel.CRITICAL,
            0.79: RiskLevel.HIGH, 0.60: RiskLevel.HIGH,
            0.59: RiskLevel.MEDIUM, 0.40: RiskLevel.MEDIUM,
            0.39: RiskLevel.LOW, 0.20: RiskLevel.LOW,
            0.19: RiskLevel.SAFE, 0.0: RiskLevel.SAFE,
        }
        for score, expected in cases.items():
            self.assertEqual(self.evaluator._determine_risk_level(score),
                             expected, f'score={score}')


class WarningMessageTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_red_and_green_light_have_hardcoded_warnings(self):
        red = _det('car', area_ratio=0.25, center=(320, 0), light_state='red')
        self.assertEqual(
            self.evaluator._evaluate_single(red, frame_width=640)
            ['warning_message'],
            '红灯，请等待，不要通行')
        green = _det('car', area_ratio=0.25, center=(320, 0), light_state='green')
        self.assertEqual(
            self.evaluator._evaluate_single(green, frame_width=640)
            ['warning_message'],
            '绿灯，请自行确认路况，不能据此判断可通行')

    def test_critical_warning_text(self):
        det = _det('car', area_ratio=0.25, center=(320, 0))
        self.assertEqual(
            self.evaluator._evaluate_single(det, frame_width=640)
            ['warning_message'],
            '危险！正前方汽车非常近，请立即停下！')

    def test_high_warning_text(self):
        # person 0.60*0.45 + 0.8*0.35 + 1.0*0.20 = 0.75 -> high
        det = _det('person', area_ratio=0.10, center=(320, 0))
        result = self.evaluator._evaluate_single(det, frame_width=640)
        self.assertEqual(result['risk_level'], 'high')
        self.assertEqual(result['warning_message'],
                         '警告！正前方有行人，距离较近，请注意避让')

    def test_medium_warning_text(self):
        det = _det('manhole_cover', area_ratio=0.04, center=(320, 0))
        self.assertEqual(
            self.evaluator._evaluate_single(det, frame_width=640)
            ['warning_message'],
            '注意，正前方有井盖')

    def test_low_warning_text(self):
        # handbag 0.15*0.45 + 0.2*0.35 + 1.0*0.20 = 0.3375 -> low
        det = _det('handbag', area_ratio=0.01, center=(320, 0))
        result = self.evaluator._evaluate_single(det, frame_width=640)
        self.assertEqual(result['risk_level'], 'low')
        self.assertEqual(result['warning_message'], '正前方检测到手提包')

    def test_safe_level_has_empty_warning(self):
        # handbag 0.15*0.45 + 0.05*0.35 + 0.3*0.20 = 0.145 -> safe
        det = _det('handbag', area_ratio=0.005, center=(0, 0))
        result = self.evaluator._evaluate_single(det, frame_width=640)
        self.assertEqual(result['risk_level'], 'safe')
        self.assertEqual(result['warning_message'], '')


class EvaluateListTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(self.evaluator.evaluate([]), [])

    def test_results_are_sorted_by_score_desc(self):
        detections = [
            _det('handbag', area_ratio=0.005, center=(0, 0)),
            _det('car', area_ratio=0.25, center=(320, 0)),
        ]
        results = self.evaluator.evaluate(detections, frame_width=640)
        self.assertEqual(results[0]['class_name'], 'car')
        self.assertGreaterEqual(results[0]['risk_score'],
                                results[1]['risk_score'])
        # 透传字段
        self.assertIn('area_ratio', results[0])
        self.assertIn('bbox', results[0])


class OverallRiskTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RiskEvaluator()

    def test_empty_results_are_safe(self):
        overall = self.evaluator.get_overall_risk([])
        self.assertEqual(overall['overall_level'], 'safe')
        self.assertEqual(overall['overall_level_cn'], '安全')
        self.assertEqual(overall['highest_score'], 0)
        self.assertEqual(overall['total_count'], 0)
        self.assertEqual(overall['summary'], '当前环境安全')

    def test_any_critical_dominates(self):
        results = self.evaluator.evaluate(
            [_det('car', area_ratio=0.25, center=(320, 0))], frame_width=640)
        overall = self.evaluator.get_overall_risk(results)
        self.assertEqual(overall['overall_level'], 'critical')
        self.assertEqual(overall['critical_count'], 1)
        self.assertEqual(overall['highest_score'], 0.955)
        self.assertEqual(overall['total_count'], 1)
        self.assertIn('极高风险', overall['summary'])

    def test_high_used_when_no_critical(self):
        results = self.evaluator.evaluate(
            [_det('person', area_ratio=0.10, center=(320, 0))], frame_width=640)
        overall = self.evaluator.get_overall_risk(results)
        self.assertEqual(overall['overall_level'], 'high')
        self.assertEqual(overall['high_count'], 1)
        self.assertEqual(overall['critical_count'], 0)
        self.assertIn('风险', overall['summary'])

    def test_falls_back_to_first_result_level_and_cn(self):
        results = self.evaluator.evaluate(
            [_det('handbag', area_ratio=0.01, center=(320, 0))],
            frame_width=640)
        overall = self.evaluator.get_overall_risk(results)
        self.assertEqual(overall['overall_level'], 'low')
        self.assertEqual(overall['overall_level_cn'], '低风险')
        self.assertIn('相对安全', overall['summary'])


if __name__ == '__main__':
    unittest.main()
