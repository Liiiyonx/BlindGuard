# -*- coding: utf-8 -*-
"""Tests for crop coordinate mapping and dual-model merging."""

import unittest

import numpy as np

from core.pipeline import DetectionPipeline, class_ids_for


class FakeDetector:
    def __init__(self, detections, names=None):
        self.detections = detections
        self.names = names or {0: 'car', 1: 'traffic_light'}
        self.seen_shapes = []
        self.seen_classes = []

    def get_class_names(self):
        return self.names

    def detect(self, frame, classes=None):
        self.seen_shapes.append(frame.shape)
        self.seen_classes.append(classes)
        return [dict(item) for item in self.detections]


class FakeRiskEvaluator:
    def evaluate(self, detections, frame_width=640):
        return [dict(item) for item in detections]

    def get_overall_risk(self, risk_results):
        return 'safe'


class FakeSceneAnalyzer:
    def analyze(self, frame, detections):
        return {'scene_type': 'test', 'count': len(detections)}


class FakeTracker:
    def update(self, detections):
        for index, detection in enumerate(detections, start=1):
            detection['track_id'] = index
        return detections


class FakeLightClassifier:
    def classify(self, frame, bbox):
        return {'state': 'red', 'score': 1.0}


def make_pipeline(detector, aux_detector=None, class_mapping=None):
    return DetectionPipeline(
        detector=detector,
        aux_detector=aux_detector,
        focus_class_ids=[0],
        aux_class_ids=None,
        portrait_crop=True,
        risk_evaluator=FakeRiskEvaluator(),
        scene_analyzer=FakeSceneAnalyzer(),
        tracker=FakeTracker(),
        light_classifier=FakeLightClassifier(),
        class_mapping=class_mapping or {'car': '汽车', 'manhole_cover': '井盖'},
    )


class PipelineTests(unittest.TestCase):
    def test_portrait_crop_maps_boxes_and_area_to_full_frame(self):
        detector = FakeDetector([
            {
                'class_name': 'car',
                'class_id': 0,
                'bbox': [100, 50, 200, 150],
                'center': (150, 100),
                'area_ratio': 0.1,
                'confidence': 0.9,
            }
        ])
        pipeline = make_pipeline(detector)
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

        result = pipeline.process(frame)

        self.assertEqual(detector.seen_shapes[0], (1344, 1080, 3))
        self.assertEqual(detector.seen_classes[0], [0])
        detection = result['detections'][0]
        self.assertEqual(detection['bbox'], [100, 242, 200, 342])
        self.assertEqual(detection['center'], (150, 292))
        self.assertAlmostEqual(detection['area_ratio'], 0.07)
        self.assertEqual(detection['class_name_cn'], '汽车')

    def test_landscape_frame_is_not_cropped(self):
        detector = FakeDetector([
            {
                'class_name': 'car',
                'class_id': 0,
                'bbox': [10, 20, 110, 120],
                'center': (60, 70),
                'area_ratio': 0.2,
                'confidence': 0.8,
            }
        ])
        pipeline = make_pipeline(detector)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = pipeline.process(frame)

        self.assertEqual(detector.seen_shapes[0], (480, 640, 3))
        detection = result['detections'][0]
        self.assertEqual(detection['bbox'], [10, 20, 110, 120])
        self.assertEqual(detection['area_ratio'], 0.2)

    def test_auxiliary_detector_adds_only_non_duplicate_classes(self):
        main = FakeDetector([
            {
                'class_name': 'car',
                'bbox': [10, 10, 100, 100],
                'center': (55, 55),
                'area_ratio': 0.2,
                'confidence': 0.9,
            }
        ])
        aux = FakeDetector(
            [
                {
                    'class_name': 'car',
                    'bbox': [12, 12, 98, 98],
                    'center': (55, 55),
                    'area_ratio': 0.18,
                    'confidence': 0.7,
                },
                {
                    'class_name': 'manhole_cover',
                    'bbox': [200, 200, 260, 260],
                    'center': (230, 230),
                    'area_ratio': 0.05,
                    'confidence': 0.8,
                },
            ],
            names={0: 'manhole_cover'},
        )
        pipeline = make_pipeline(main, aux_detector=aux)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = pipeline.process(frame)

        self.assertEqual(
            [item['class_name'] for item in result['detections']],
            ['car', 'manhole_cover'],
        )
        self.assertEqual(aux.seen_shapes[0], (480, 640, 3))

    def test_class_ids_for_maps_only_known_names(self):
        detector = FakeDetector([], names={0: 'car', 1: 'person'})
        self.assertEqual(class_ids_for(detector, ['person', 'missing']), [1])
        self.assertIsNone(class_ids_for(detector, []))


if __name__ == '__main__':
    unittest.main()
