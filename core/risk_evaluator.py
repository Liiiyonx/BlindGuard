# -*- coding: utf-8 -*-
"""
风险评估模块
根据检测结果评估环境风险等级

本模块实现了多维度风险评估算法，
综合考虑目标类型、距离、位置等因素，
为视障用户提供准确的风险预警。
"""

import logging
from typing import List, Dict, Optional
from enum import Enum

from core.geometry import POSITION_CENTER_ZONE, position_zone
from core.labels import get_class_cn

logger = logging.getLogger('BlindGuard.Risk')


class RiskLevel(Enum):
    """风险等级枚举"""
    CRITICAL = 'critical'   # 极高风险 - 立即停止
    HIGH = 'high'          # 高风险 - 需要立即避让
    MEDIUM = 'medium'      # 中等风险 - 需要注意
    LOW = 'low'            # 低风险 - 需要留意
    SAFE = 'safe'          # 安全


class RiskEvaluator:
    """
    风险评估器
    基于多因素综合评估环境风险
    """

    # 风险等级权重配置（三项权重之和为 1.0，保证总分上限为 1）
    # 此前定义了 movement_risk: 0.15 但从未参与计算，导致总分上限只有 0.85，
    # "极高风险"(>=0.80) 几乎无法触发；运动趋势现由 tracker 提供给 Agent 使用
    RISK_WEIGHTS = {
        'class_risk': 0.45,      # 目标类别风险权重
        'distance_risk': 0.35,   # 距离风险权重
        'position_risk': 0.20    # 位置风险权重
    }

    # 目标类别基础风险分值 (0-1)
    #
    # 优先级：本表 > _high/_medium/_low_risk_classes 三个集合（0.90/0.60/0.30）。
    # 因此只需登记「需要单独定分」的类别；集合内的类别由集合统一分档，
    # 在此重复登记会掩盖集合配置（历史上 car/truck/person 等条目即被集合覆盖而失效）。
    # 可通过 config 的 risk.class_scores 覆盖本表。
    CLASS_RISK_SCORES = {
        # 名称与主模型类别表不一致的写法，保留以兼容不同数据集的标签
        'traffic light': 0.50,
        'pole': 0.40,
        'handbag': 0.15,
        'bird': 0.20,

        # 默认值：三个集合与上表都未命中的类别
        'default': 0.30
    }

    # 距离等级阈值 (基于面积占比)
    DISTANCE_THRESHOLDS = {
        'very_close': 0.20,   # 非常近 (>20% 画面)
        'close': 0.10,        # 近 (10-20% 画面)
        'medium': 0.04,       # 中等 (4-10% 画面)
        'far': 0.01,          # 远 (1-4% 画面)
        'very_far': 0.0       # 非常远 (<1% 画面)
    }

    # 位置区域定义 (基于画面宽度比例)
    POSITION_ZONES = {
        'center': POSITION_CENTER_ZONE,   # 中心区域
        'left_near': (0.0, 0.25),         # 左侧近区
        'right_near': (0.75, 1.0),        # 右侧近区
    }

    def __init__(self, risk_thresholds: Optional[Dict] = None,
                 high_risk_classes: Optional[List[str]] = None,
                 medium_risk_classes: Optional[List[str]] = None,
                 low_risk_classes: Optional[List[str]] = None,
                 class_mapping: Optional[Dict] = None):
        """
        初始化风险评估器

        Args:
            risk_thresholds: 自定义风险阈值配置
            high_risk_classes: 高风险类别列表
            medium_risk_classes: 中风险类别列表
            low_risk_classes: 低风险类别列表
            class_mapping: 类别英文->中文映射（来自 config.yaml class_mapping）
        """
        self._class_mapping = dict(class_mapping) if class_mapping else {}

        # 实例级阈值副本（关键：避免污染类属性）
        # CLASS_RISK_SCORES / DISTANCE_THRESHOLDS 是类属性，若直接 update
        # 会改动类本身，污染所有现存及未来实例（一处自定义阈值，全局生效）。
        # 这里先浅拷贝为实例属性（遮蔽同名类属性），后续自定义只改实例副本；
        # 值均为不可变标量，浅拷贝足够。默认实例（无自定义阈值）行为完全不变。
        self.CLASS_RISK_SCORES = dict(type(self).CLASS_RISK_SCORES)
        self.DISTANCE_THRESHOLDS = dict(type(self).DISTANCE_THRESHOLDS)

        # 应用自定义阈值
        if risk_thresholds:
            self._apply_custom_thresholds(risk_thresholds)

        # 应用自定义风险类别
        if high_risk_classes:
            self._high_risk_classes = set(high_risk_classes)
        else:
            self._high_risk_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'fire hydrant'}

        if medium_risk_classes:
            self._medium_risk_classes = set(medium_risk_classes)
        else:
            self._medium_risk_classes = {'person', 'dog', 'cat', 'chair', 'bench', 'potted plant', 'suitcase', 'backpack', 'umbrella', 'stop sign'}

        if low_risk_classes:
            self._low_risk_classes = set(low_risk_classes)
        else:
            self._low_risk_classes = {'bottle', 'cup', 'book', 'cell phone', 'laptop', 'mouse', 'keyboard', 'tv', 'remote', 'clock'}

        # 风险等级中文映射
        self.level_names = {
            RiskLevel.CRITICAL: '极高风险',
            RiskLevel.HIGH: '高风险',
            RiskLevel.MEDIUM: '中等风险',
            RiskLevel.LOW: '低风险',
            RiskLevel.SAFE: '安全'
        }

        logger.info(f"风险评估器初始化完成")
        logger.info(f"高风险类别: {self._high_risk_classes}")
        logger.info(f"中风险类别: {self._medium_risk_classes}")

    def _apply_custom_thresholds(self, thresholds: Dict):
        """应用自定义阈值配置

        只更新本实例的阈值副本（__init__ 中已从类属性浅拷贝），
        不会改动类属性，因此不会影响其它现存或未来实例。
        """
        # 更新类别风险分值
        if 'class_scores' in thresholds:
            self.CLASS_RISK_SCORES.update(thresholds['class_scores'])

        # 更新距离阈值
        if 'distance' in thresholds:
            self.DISTANCE_THRESHOLDS.update(thresholds['distance'])

    def evaluate(self, detections: List[Dict],
                 frame_width: int = 640) -> List[Dict]:
        """
        评估检测结果的风险

        Args:
            detections: 检测结果列表
            frame_width: 当前帧宽度（像素），用于方位/位置判断

        Returns:
            风险评估结果列表，每个结果包含:
            - risk_level: 风险等级
            - risk_score: 风险分值 (0-1)
            - risk_factors: 各风险因素详情
            - warning_message: 警告消息
            - distance_level: 距离等级
            - position_zone: 位置区域
        """
        if not detections:
            return []

        risk_results = []

        for det in detections:
            risk_result = self._evaluate_single(det, frame_width)
            risk_results.append(risk_result)

        # 按风险分值排序
        risk_results.sort(key=lambda x: x['risk_score'], reverse=True)

        return risk_results

    def _evaluate_single(self, detection: Dict, frame_width: int = 640) -> Dict:
        """评估单个检测结果的风险"""
        # 提取检测信息
        class_name = detection.get('class_name', 'default')
        area_ratio = detection.get('area_ratio', 0)
        center = detection.get('center', (0, 0))

        # 计算各维度风险分值
        class_score = self._calculate_class_risk(class_name)
        distance_score = self._calculate_distance_risk(area_ratio)
        position_score = self._calculate_position_risk(center[0], frame_width)

        # 计算综合风险分值
        total_score = (
            class_score * self.RISK_WEIGHTS['class_risk'] +
            distance_score * self.RISK_WEIGHTS['distance_risk'] +
            position_score * self.RISK_WEIGHTS['position_risk']
        )

        # 红灯亮起：过马路场景的硬约束，额外加权
        light_state = detection.get('light_state')
        if light_state == 'red':
            total_score = min(1.0, total_score + 0.15)

        # 确定风险等级
        risk_level = self._determine_risk_level(total_score)

        # 确定距离等级
        distance_level = self._determine_distance_level(area_ratio)

        # 确定位置区域
        position_zone = self._determine_position_zone(center[0], frame_width)

        # 生成警告消息
        warning_message = self._generate_warning(
            class_name, risk_level, distance_level, position_zone, light_state
        )

        result = {
            'class_name': class_name,
            'risk_level': risk_level.value,
            'risk_level_cn': self.level_names[risk_level],
            'risk_score': round(total_score, 3),
            'risk_factors': {
                'class_score': round(class_score, 3),
                'distance_score': round(distance_score, 3),
                'position_score': round(position_score, 3)
            },
            'distance_level': distance_level,
            'position_zone': position_zone,
            'warning_message': warning_message,
            'bbox': detection.get('bbox', []),
            'confidence': detection.get('confidence', 0),
            'area_ratio': area_ratio
        }
        # 透传跟踪器/红绿灯的附加信息
        for key in ('light_state', 'track_id', 'track_age', 'trend',
                    'conf_stable'):
            if key in detection:
                result[key] = detection[key]
        return result

    def _calculate_class_risk(self, class_name: str) -> float:
        """
        计算目标类别风险分值
        优先使用显式登记的分值（可被 config 覆盖），否则按风险集合分档
        """
        # 显式分值优先，保证 risk.class_scores 的覆盖真实生效
        explicit = self.CLASS_RISK_SCORES.get(class_name)
        if explicit is not None:
            return explicit

        # 高风险类别
        if class_name in self._high_risk_classes:
            return 0.90

        # 中风险类别
        if class_name in self._medium_risk_classes:
            return 0.60

        # 低风险类别
        if class_name in self._low_risk_classes:
            return 0.30

        # 使用默认分值
        return self.CLASS_RISK_SCORES['default']

    def _calculate_distance_risk(self, area_ratio: float) -> float:
        """
        计算距离风险分值
        基于目标在画面中的面积占比判断距离
        """
        if area_ratio >= self.DISTANCE_THRESHOLDS['very_close']:
            return 1.0  # 极近
        elif area_ratio >= self.DISTANCE_THRESHOLDS['close']:
            return 0.8  # 近
        elif area_ratio >= self.DISTANCE_THRESHOLDS['medium']:
            return 0.5  # 中等
        elif area_ratio >= self.DISTANCE_THRESHOLDS['far']:
            return 0.2  # 远
        else:
            return 0.05  # 极远

    def _calculate_position_risk(self, center_x: float,
                                  frame_width: int) -> float:
        """
        计算位置风险分值
        位于画面中心的目标风险更高
        """
        # 归一化x坐标到0-1范围
        normalized_x = center_x / frame_width

        # 中心区域风险最高
        center_start, center_end = self.POSITION_ZONES['center']

        if center_start <= normalized_x <= center_end:
            # 距离中心越近风险越高
            center_distance = abs(normalized_x - 0.5) * 2  # 0-1
            return 1.0 - center_distance * 0.5  # 0.5-1.0
        else:
            # 侧面区域风险较低
            return 0.3

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """根据分值确定风险等级"""
        if score >= 0.80:
            return RiskLevel.CRITICAL
        elif score >= 0.60:
            return RiskLevel.HIGH
        elif score >= 0.40:
            return RiskLevel.MEDIUM
        elif score >= 0.20:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE

    def _determine_distance_level(self, area_ratio: float) -> str:
        """确定距离等级"""
        if area_ratio >= self.DISTANCE_THRESHOLDS['very_close']:
            return 'very_close'
        elif area_ratio >= self.DISTANCE_THRESHOLDS['close']:
            return 'close'
        elif area_ratio >= self.DISTANCE_THRESHOLDS['medium']:
            return 'medium'
        elif area_ratio >= self.DISTANCE_THRESHOLDS['far']:
            return 'far'
        else:
            return 'very_far'

    def _determine_position_zone(self, center_x: float,
                                  frame_width: int) -> str:
        """确定位置区域"""
        return position_zone(center_x / frame_width,
                             self.POSITION_ZONES['center'])

    def _generate_warning(self, class_name: str, risk_level: RiskLevel,
                          distance_level: str, position_zone: str,
                          light_state: Optional[str] = None) -> str:
        """
        生成警告消息

        综合考虑目标类型、风险等级、距离、位置与红绿灯状态，
        生成自然语言警告消息。
        """
        # 距离描述
        distance_desc = {
            'very_close': '非常近',
            'close': '较近',
            'medium': '中等距离',
            'far': '较远',
            'very_far': '很远'
        }

        # 位置描述
        position_desc = {
            'center': '正前方',
            'left': '左侧',
            'right': '右侧'
        }

        cn_name = get_class_cn(class_name, self._class_mapping)
        dist_desc = distance_desc.get(distance_level, '')
        pos_desc = position_desc.get(position_zone, '')

        # 红灯：给出通行约束而非障碍警告
        if light_state == 'red':
            return "红灯，请等待，不要通行"
        if light_state == 'green':
            return "绿灯，请自行确认路况，不能据此判断可通行"

        # 根据风险等级生成不同紧急程度的警告
        if risk_level == RiskLevel.CRITICAL:
            return f"危险！{pos_desc}{cn_name}{dist_desc}，请立即停下！"
        elif risk_level == RiskLevel.HIGH:
            return f"警告！{pos_desc}有{cn_name}，距离{dist_desc}，请注意避让"
        elif risk_level == RiskLevel.MEDIUM:
            return f"注意，{pos_desc}有{cn_name}"
        elif risk_level == RiskLevel.LOW:
            return f"{pos_desc}检测到{cn_name}"
        else:
            return ""

    def get_overall_risk(self, risk_results: List[Dict]) -> Dict:
        """
        获取整体风险评估

        Args:
            risk_results: 风险评估结果列表

        Returns:
            整体风险评估结果
        """
        if not risk_results:
            return {
                'overall_level': 'safe',
                'overall_level_cn': '安全',
                'highest_score': 0,
                'critical_count': 0,
                'high_count': 0,
                'total_count': 0,
                'summary': '当前环境安全'
            }

        # 统计各等级数量
        critical_count = sum(1 for r in risk_results if r['risk_level'] == 'critical')
        high_count = sum(1 for r in risk_results if r['risk_level'] == 'high')

        # 确定整体风险等级（取最高等级）
        if critical_count > 0:
            overall_level = 'critical'
        elif high_count > 0:
            overall_level = 'high'
        else:
            overall_level = risk_results[0]['risk_level']

        # 获取最高风险分值
        highest_score = max(r['risk_score'] for r in risk_results)

        # 生成摘要
        summary = self._generate_summary(risk_results, overall_level)

        return {
            'overall_level': overall_level,
            'overall_level_cn': self.level_names.get(
                RiskLevel(overall_level), '未知'
            ),
            'highest_score': round(highest_score, 3),
            'critical_count': critical_count,
            'high_count': high_count,
            'total_count': len(risk_results),
            'summary': summary
        }

    def _generate_summary(self, risk_results: List[Dict],
                          overall_level: str) -> str:
        """生成风险摘要"""
        if not risk_results:
            return "当前环境安全"

        total = len(risk_results)
        critical = sum(1 for r in risk_results if r['risk_level'] == 'critical')
        high = sum(1 for r in risk_results if r['risk_level'] == 'high')

        if critical > 0:
            return f"检测到{total}个目标，其中{critical}个极高风险，请立即注意安全"
        elif high > 0:
            return f"检测到{total}个目标，{high}个高风险，请注意周围环境"
        else:
            return f"检测到{total}个目标，整体环境相对安全"
