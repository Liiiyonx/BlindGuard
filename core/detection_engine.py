# -*- coding: utf-8 -*-
"""
检测引擎模块
负责目标检测模型的加载和推理

本模块实现了基于YOLO的实时目标检测功能，
支持多种模型格式和硬件加速。
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    from ultralytics import YOLOv10 as YOLOModel
except ImportError:
    from ultralytics import YOLO as YOLOModel

logger = logging.getLogger('BlindGuard.Detection')


class DetectionEngine:
    """
    目标检测引擎
    封装YOLO模型，提供统一的检测接口
    """

    # 默认检测类别（针对导盲场景优化）
    PRIORITY_CLASSES = {
        'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
        'dog', 'cat', 'traffic light', 'stop sign', 'fire hydrant',
        'bench', 'chair', 'umbrella', 'handbag', 'suitcase'
    }

    def __init__(self, model_path: str = 'best.pt',
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = '',
                 img_size: int = 640):
        """
        初始化检测引擎

        Args:
            model_path: 模型文件路径（默认使用训练好的best.pt）
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 计算设备 ('cpu', 'cuda', '')
            img_size: 推理图像尺寸
        """
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size

        self.model = None
        self.class_names = {}
        self.is_loaded = False

        # 性能统计
        self.inference_times = []
        self.total_frames = 0

        # 自动加载模型
        self._load_model()

    def _load_model(self):
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            start_time = time.time()

            self.model = YOLOModel(self.model_path)

            # 指定设备
            if self.device:
                self.model.to(self.device)

            # 获取类别名称
            self.class_names = self.model.names

            load_time = time.time() - start_time
            self.is_loaded = True

            logger.info(f"模型加载成功，耗时: {load_time:.2f}秒")
            logger.info(f"检测类别数: {len(self.class_names)}")
            logger.info(f"类别列表: {list(self.class_names.values())}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.is_loaded = False
            raise

    def detect(self, frame: np.ndarray,
               classes: Optional[List[int]] = None) -> List[Dict]:
        """
        执行目标检测

        Args:
            frame: 输入图像 (BGR格式)
            classes: 指定检测的类别ID列表，None表示检测所有类别

        Returns:
            检测结果列表，每个结果包含:
            - bbox: 边界框 [x1, y1, x2, y2]
            - class_name: 类别名称
            - class_id: 类别ID
            - confidence: 置信度
            - center: 中心点坐标 (x, y)
            - area: 面积
        """
        if not self.is_loaded:
            logger.warning("模型未加载，无法执行检测")
            return []

        if frame is None or frame.size == 0:
            return []

        start_time = time.time()

        try:
            # 执行推理
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes,
                verbose=False
            )

            # 解析结果
            detections = self._parse_results(results, frame.shape)

            # 更新性能统计
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            self.total_frames += 1

            return detections

        except Exception as e:
            logger.error(f"检测推理失败: {e}")
            return []

    def _parse_results(self, results, frame_shape: Tuple) -> List[Dict]:
        """解析模型输出"""
        detections = []
        frame_h, frame_w = frame_shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 提取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 提取类别和置信度
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f'class_{class_id}')

                # 计算中心点和面积
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # 边界检查
                x1 = max(0, min(x1, frame_w))
                y1 = max(0, min(y1, frame_h))
                x2 = max(0, min(x2, frame_w))
                y2 = max(0, min(y2, frame_h))

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': round(confidence, 4),
                    'center': (round(center_x, 1), round(center_y, 1)),
                    'width': width,
                    'height': height,
                    'area': int(area),
                    'area_ratio': round(area / (frame_w * frame_h), 6)
                }

                detections.append(detection)

        return detections

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"置信度阈值更新为: {self.conf_threshold}")

    def set_iou_threshold(self, threshold: float):
        """设置IoU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"IoU阈值更新为: {self.iou_threshold}")

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'fps': 0,
                'total_frames': self.total_frames
            }

        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_inference_time': round(avg_time * 1000, 2),  # 毫秒
            'fps': round(fps, 1),
            'total_frames': self.total_frames
        }

    def get_class_names(self) -> Dict[int, str]:
        """获取所有类别名称"""
        return dict(self.class_names)

    def reload_model(self, model_path: Optional[str] = None):
        """重新加载模型"""
        if model_path:
            self.model_path = model_path
        self._load_model()
