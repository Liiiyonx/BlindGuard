# -*- coding: utf-8 -*-
"""
红绿灯状态识别（纯 CV，零 API 成本）

YOLO 只能检测出"画面里有一个红绿灯"，看不出亮的是哪个灯——
而灯的状态恰恰是视障人士过马路的刚需信息。
本模块对检测框内区域做 HSV 颜色统计，判断红/黄/绿状态，
供风险评估（红灯加权）与智能体（"红灯，请等待"）使用。

说明：这是启发式方法，强逆光、遮挡、非标准灯色下可能返回 unknown，
属预期兜底行为；如需更高准确率可改接多模态模型（agent 的 look_at_frame 工具）。
"""

import logging

import cv2

logger = logging.getLogger('BlindGuard.TrafficLight')

STATE_CN = {
    'red': '红灯',
    'yellow': '黄灯',
    'green': '绿灯',
    'unknown': '灯色未知',
}

# HSV 颜色范围默认值：名称 -> [(h_lo, h_hi, s_min, v_min), ...]（OpenCV H 取值 0-180）
# 红色横跨 H=0 两侧，需要两段区间；实际参数为实例属性，可被调参脚本基于真实数据优化
# 当前值由 tune_light.py 在 BDD 1884 个真实红绿灯上网格搜索得出
# （留出测试集 74.0% -> 79.3%），黄/红 hue 分离(20-40 与 0-13/177-180)是关键
DEFAULT_COLOR_RANGES = {
    'red': [
        (0, 13, 60, 70),
        (177, 180, 60, 70),
    ],
    'yellow': [
        (20, 40, 60, 70),
    ],
    'green': [
        (40, 90, 40, 60),
    ],
}


class TrafficLightClassifier:
    """基于 HSV 颜色统计的红绿灯状态分类器"""

    def __init__(self, min_pixels: int = 10, color_ranges: dict = None):
        """
        Args:
            min_pixels: 判定为"点亮"所需的最少符合像素数，
                        低于该值视为无法识别（unknown）
            color_ranges: 自定义 HSV 颜色范围（默认用 DEFAULT_COLOR_RANGES），
                          可由调参脚本基于真实数据优化
        """
        self.min_pixels = max(1, int(min_pixels))
        import copy
        self._ranges = copy.deepcopy(color_ranges or DEFAULT_COLOR_RANGES)

    def classify(self, frame, bbox) -> dict:
        """
        判断一个红绿灯检测框的亮灯状态

        Args:
            frame: BGR 图像
            bbox: 检测框 [x1, y1, x2, y2]

        Returns:
            {'state': 'red'|'yellow'|'green'|'unknown', 'score': 占比}
            score 为命中像素占框面积的比例，可用于置信度参考
        """
        if frame is None or not bbox or len(bbox) < 4:
            return {'state': 'unknown', 'score': 0.0}

        fh, fw = frame.shape[:2]
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except (TypeError, ValueError):
            return {'state': 'unknown', 'score': 0.0}
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return {'state': 'unknown', 'score': 0.0}

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        best_state, best_count = 'unknown', 0
        for state, ranges in self._ranges.items():
            count = 0
            for h_lo, h_hi, s_min, v_min in ranges:
                mask = (hue >= h_lo) & (hue <= h_hi) & (sat >= s_min) & (val >= v_min)
                count += int(mask.sum())
            if count > best_count:
                best_state, best_count = state, count

        if best_count >= self.min_pixels:
            area = roi.shape[0] * roi.shape[1]
            return {'state': best_state, 'score': round(best_count / area, 4)}
        return {'state': 'unknown', 'score': 0.0}
