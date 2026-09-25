# -*- coding: utf-8 -*-
"""Production detection pipeline shared by the web app and evaluation tools."""

import time
from typing import Dict, Iterable, List, Optional

from core.geometry import box_iou
from core.labels import TRAFFIC_LIGHT_CLASSES
from core.traffic_light import STATE_CN


def class_ids_for(detector, class_names: Optional[Iterable[str]]) -> Optional[List[int]]:
    """Map configured class names to model IDs."""
    if detector is None or not class_names:
        return None
    name_to_id = {name: class_id for class_id, name in detector.get_class_names().items()}
    ids = [name_to_id[name] for name in class_names if name in name_to_id]
    return ids or None


class DetectionPipeline:
    """Run crop, main/aux detection, light classification, tracking, and risk."""

    def __init__(self, detector, risk_evaluator, scene_analyzer, tracker,
                 light_classifier, aux_detector=None,
                 focus_class_ids: Optional[List[int]] = None,
                 aux_class_ids: Optional[List[int]] = None,
                 portrait_crop: bool = False,
                 class_mapping: Optional[Dict] = None):
        self.detector = detector
        self.aux_detector = aux_detector
        self.focus_class_ids = focus_class_ids
        self.aux_class_ids = aux_class_ids
        self.portrait_crop = bool(portrait_crop)
        self.risk = risk_evaluator
        self.scene = scene_analyzer
        self.tracker = tracker
        self.light_classifier = light_classifier
        self.class_mapping = dict(class_mapping) if class_mapping else {}

    def process(self, frame) -> Dict:
        """Process one BGR frame without mutating it."""
        started = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]

        crop_started = time.perf_counter()
        crop_top, crop_bot = 0, frame_h
        do_crop = self.portrait_crop and frame_h > frame_w * 1.3
        if do_crop:
            crop_top = int(frame_h * 0.10)
            crop_bot = int(frame_h * 0.80)
        detect_img = frame[crop_top:crop_bot, :] if do_crop else frame
        crop_ms = (time.perf_counter() - crop_started) * 1000

        main_started = time.perf_counter()
        detections = self.detector.detect(
            detect_img, classes=self.focus_class_ids)
        if do_crop:
            self._map_crop_coordinates(
                detections, crop_top, crop_bot - crop_top, frame_h)
        main_ms = (time.perf_counter() - main_started) * 1000

        aux_ms = 0.0
        if self.aux_detector is not None:
            aux_started = time.perf_counter()
            aux_detections = self.aux_detector.detect(
                frame, classes=self.aux_class_ids)
            aux_ms = (time.perf_counter() - aux_started) * 1000
            detections = self._merge_detections(detections, aux_detections)

        light_started = time.perf_counter()
        for det in detections:
            if det.get("class_name") in TRAFFIC_LIGHT_CLASSES:
                det["light_state"] = self.light_classifier.classify(
                    frame, det.get("bbox"))["state"]
        light_ms = (time.perf_counter() - light_started) * 1000

        track_started = time.perf_counter()
        self.tracker.update(detections)
        track_ms = (time.perf_counter() - track_started) * 1000

        risk_started = time.perf_counter()
        risk_results = self.risk.evaluate(detections, frame_width=frame_w)
        enriched = self.enrich_detections(risk_results)
        overall = self.risk.get_overall_risk(risk_results)
        risk_ms = (time.perf_counter() - risk_started) * 1000

        scene_started = time.perf_counter()
        scene_info = self.scene.analyze(frame, detections)
        scene_ms = (time.perf_counter() - scene_started) * 1000

        return {
            "detections": enriched,
            "risk_results": risk_results,
            "overall": overall,
            "scene": scene_info,
            "frame_size": (frame_w, frame_h),
            "timings": {
                "crop_ms": crop_ms,
                "main_detect_ms": main_ms,
                "aux_detect_ms": aux_ms,
                "light_ms": light_ms,
                "track_ms": track_ms,
                "risk_ms": risk_ms,
                "scene_ms": scene_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }

    def enrich_detections(self, risk_results: List[Dict]) -> List[Dict]:
        """Add Chinese class/light labels expected by the UI and Agent."""
        enriched = []
        for result in risk_results:
            det = dict(result)
            det["class_name_cn"] = self.class_mapping.get(
                det["class_name"], det["class_name"])
            if det.get("light_state"):
                det["light_state_cn"] = STATE_CN.get(
                    det["light_state"], det["light_state"])
            enriched.append(det)
        return enriched

    @staticmethod
    def _map_crop_coordinates(detections: List[Dict], crop_top: int,
                              crop_height: int, frame_height: int) -> None:
        """Map crop-space boxes and areas back to the full frame."""
        area_scale = crop_height / max(1, frame_height)
        for det in detections:
            bbox = det.get("bbox") or []
            if len(bbox) >= 4:
                det["bbox"] = [
                    bbox[0], bbox[1] + crop_top,
                    bbox[2], bbox[3] + crop_top,
                ]
            center = det.get("center")
            if center:
                det["center"] = (center[0], center[1] + crop_top)
            det["area_ratio"] = det.get("area_ratio", 0) * area_scale

    def _merge_detections(self, main_detections: List[Dict],
                          aux_detections: List[Dict]) -> List[Dict]:
        """Keep main-model boxes first and add non-duplicate aux detections."""
        merged = list(main_detections)
        for aux in aux_detections:
            duplicate = any(
                main.get("class_name") == aux.get("class_name")
                and box_iou(main.get("bbox", []), aux.get("bbox", [])) > 0.55
                for main in main_detections
            )
            if not duplicate:
                merged.append(aux)
        return merged
