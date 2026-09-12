# -*- coding: utf-8 -*-
"""
轻量目标跟踪（IoU 匹配，无外部依赖）

为每帧检测结果分配稳定的 track_id，并基于面积变化估计运动趋势：
- approaching：目标面积比约 1 秒前明显增大（正在接近）
- receding：明显减小（正在远离）
- stable：基本不变

有了跟踪信息，播报才能从"前方有汽车"升级为"左后方汽车正在接近"。
采用"类别内 IoU 贪心匹配"的简单策略：导盲场景目标数少、帧率高，
足够使用且确定性强，不引入滤波器等重依赖。
"""

import time
from collections import deque

TREND_CN = {
    'approaching': '正在接近',
    'receding': '正在远离',
    'stable': '',
}


def _iou(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """类别内 IoU 贪心匹配的多目标跟踪器"""

    def __init__(self, iou_threshold: float = 0.3, max_missing: float = 1.0,
                 trend_window: float = 1.0, trend_ratio: float = 1.25):
        """
        Args:
            iou_threshold: 匹配判定阈IoU 下限
            max_missing: 轨迹失联多少秒后删除
            trend_window: 计算趋势时回看的秒数
            trend_ratio: 面积变化超过该倍数视为接近，低于其倒数视为远离
        """
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.trend_window = trend_window
        self.trend_ratio = trend_ratio

        self._tracks = {}          # id -> 轨迹字典
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def update(self, detections):
        """
        为一帧检测结果标注跟踪信息（就地修改 det 字典）：
        - track_id: int，跨帧稳定
        - track_age: float，目标已持续出现的秒数
        - trend: 'approaching' | 'receding' | 'stable'
        """
        now = time.time()
        matched_ids = set()

        for det in detections:
            cls = det.get('class_name', '')
            bbox = list(det.get('bbox') or [0, 0, 0, 0])
            if len(bbox) < 4:
                bbox = [0, 0, 0, 0]
            area = float(det.get('area') or max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            # 在同类轨迹中找 IoU 最大的未匹配轨迹
            best, best_iou = None, self.iou_threshold
            for tr in self._tracks.values():
                if tr['class_name'] != cls or tr['id'] in matched_ids:
                    continue
                iou = _iou(bbox, tr['bbox'])
                if iou > best_iou:
                    best, best_iou = tr, iou

            if best is None:
                best = {
                    'id': self._next_id,
                    'class_name': cls,
                    'bbox': bbox,
                    'first_seen': now,
                    'last_seen': now,
                    'history': deque(maxlen=128),  # (timestamp, area)
                }
                self._next_id += 1
                self._tracks[best['id']] = best

            matched_ids.add(best['id'])
            best['bbox'] = bbox
            best['last_seen'] = now
            best['history'].append((now, area))

            det['track_id'] = best['id']
            det['track_age'] = round(now - best['first_seen'], 1)

            # 趋势：与 trend_window 秒前最早的样本比较面积
            ref_area = None
            for t, a in best['history']:
                if now - t >= self.trend_window:
                    ref_area = a
                    break
            if ref_area and ref_area > 0:
                ratio = area / ref_area
                if ratio >= self.trend_ratio:
                    det['trend'] = 'approaching'
                elif ratio <= 1.0 / self.trend_ratio:
                    det['trend'] = 'receding'
                else:
                    det['trend'] = 'stable'
            else:
                det['trend'] = 'stable'

        # 清理失联轨迹
        expired = [tid for tid, tr in self._tracks.items()
                   if now - tr['last_seen'] > self.max_missing]
        for tid in expired:
            del self._tracks[tid]

        return detections

    def reset(self):
        """清空所有轨迹（切换视频源时调用）"""
        self._tracks.clear()
