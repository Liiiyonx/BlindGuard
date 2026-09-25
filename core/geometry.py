# -*- coding: utf-8 -*-
"""
几何判定共用实现

IoU 与画面方位分带此前在 tracker、pipeline、evaluate、risk_evaluator、
scene_analyzer、agent 各自实现了一份。同一判定散落多处，改一处漏一处会
静默产生不一致（例如合并去重与跟踪匹配用了不同的 IoU），故收敛到这里。
"""

# 画面方位分带：左带 < start，右带 > end，中间为闭区间 [start, end]
POSITION_CENTER_ZONE = (0.25, 0.75)


def box_iou(a, b) -> float:
    """两个 [x1, y1, x2, y2] 框的 IoU；无交集或退化框返回 0.0"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def position_zone(normalized_x: float,
                  center_zone=POSITION_CENTER_ZONE) -> str:
    """按归一化横坐标返回 'left' / 'center' / 'right'。

    center_zone 为中间带的 (start, end) 闭区间，端点归中间带。
    """
    start, end = center_zone
    if start <= normalized_x <= end:
        return 'center'
    return 'left' if normalized_x < start else 'right'
