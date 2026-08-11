# -*- coding: utf-8 -*-
"""
场景分析模块
对检测结果进行场景级理解和分析

本模块实现了环境场景的综合分析功能，
包括目标分布统计、空间关系分析和场景类型判断，
为风险评估提供更全面的上下文信息。
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger('BlindGuard.Scene')


class SceneAnalyzer:
    """
    场景分析器
    分析检测结果的场景特征和空间关系
    """

    # 场景类型定义
    SCENE_TYPES = {
        'empty': '空旷场景',
        'simple': '简单场景',
        'complex': '复杂场景',
        'crowded': '拥挤场景',
        'dangerous': '危险场景'
    }

    def __init__(self):
        """初始化场景分析器"""
        # 目标类别分组
        self.category_groups = {
            'vehicle': {'car', 'truck', 'bus', 'motorcycle', 'bicycle'},
            'pedestrian': {'person'},
            'traffic_sign': {'traffic light', 'stop sign', 'parking meter'},
            'obstacle': {'fire hydrant', 'bench', 'chair', 'pole'},
            'animal': {'dog', 'cat', 'bird', 'horse'},
            'movable': {'umbrella', 'handbag', 'suitcase', 'backpack'}
        }

        # 上一帧分析结果（用于运动分析）
        self.prev_analysis = None

        logger.info("场景分析器初始化完成")

    def analyze(self, frame: Optional[any],
                detections: List[Dict]) -> Dict:
        """
        分析当前场景

        Args:
            frame: 输入图像
            detections: 检测结果列表

        Returns:
            场景分析结果，包含:
            - scene_type: 场景类型
            - object_distribution: 目标分布统计
            - spatial_analysis: 空间分析
            - movement_analysis: 运动分析
            - summary: 场景摘要
        """
        if not detections:
            return self._empty_scene_result()

        # 目标分布分析
        distribution = self._analyze_distribution(detections)

        # 空间分析
        spatial = self._analyze_spatial(detections)

        # 运动分析
        movement = self._analyze_movement(detections)

        # 确定场景类型
        scene_type = self._determine_scene_type(detections, distribution)

        # 生成场景摘要
        summary = self._generate_summary(
            scene_type, distribution, spatial, movement
        )

        result = {
            'scene_type': scene_type,
            'scene_type_cn': self.SCENE_TYPES.get(scene_type, '未知场景'),
            'object_distribution': distribution,
            'spatial_analysis': spatial,
            'movement_analysis': movement,
            'summary': summary,
            'object_count': len(detections),
            'timestamp': self._get_timestamp()
        }

        # 保存当前分析结果
        self.prev_analysis = result

        return result

    def _empty_scene_result(self) -> Dict:
        """空场景结果"""
        return {
            'scene_type': 'empty',
            'scene_type_cn': '空旷场景',
            'object_distribution': {},
            'spatial_analysis': {'left': 0, 'center': 0, 'right': 0},
            'movement_analysis': {},
            'summary': '当前视野内无明显障碍物',
            'object_count': 0,
            'timestamp': self._get_timestamp()
        }

    def _analyze_distribution(self, detections: List[Dict]) -> Dict:
        """
        分析目标分布
        统计各类目标的数量和占比
        """
        # 统计各类别数量
        class_counter = Counter(d['class_name'] for d in detections)

        # 按类别分组统计
        category_counter = Counter()
        for class_name, count in class_counter.items():
            category = self._get_category(class_name)
            category_counter[category] += count

        # 计算占比
        total = len(detections)
        class_distribution = {
            name: {
                'count': count,
                'ratio': round(count / total, 3)
            }
            for name, count in class_counter.most_common()
        }

        category_distribution = {
            name: {
                'count': count,
                'ratio': round(count / total, 3)
            }
            for name, count in category_counter.most_common()
        }

        return {
            'by_class': class_distribution,
            'by_category': category_distribution,
            'total_count': total
        }

    def _analyze_spatial(self, detections: List[Dict]) -> Dict:
        """
        空间分析
        分析目标在画面中的位置分布
        """
        # 位置统计
        left_count = 0
        center_count = 0
        right_count = 0

        for det in detections:
            center_x = det.get('center', (0, 0))[0]
            frame_width = 640  # 默认帧宽度

            # 计算归一化位置
            normalized_x = center_x / frame_width

            if normalized_x < 0.25:
                left_count += 1
            elif normalized_x < 0.75:
                center_count += 1
            else:
                right_count += 1

        total = len(detections)

        return {
            'left': left_count,
            'center': center_count,
            'right': right_count,
            'left_ratio': round(left_count / total, 3) if total > 0 else 0,
            'center_ratio': round(center_count / total, 3) if total > 0 else 0,
            'right_ratio': round(right_count / total, 3) if total > 0 else 0,
            'dominant_side': self._get_dominant_side(left_count, center_count, right_count)
        }

    def _analyze_movement(self, detections: List[Dict]) -> Dict:
        """
        运动分析
        通过对比前后帧分析目标运动状态
        """
        if self.prev_analysis is None:
            return {
                'new_objects': len(detections),
                'disappeared_objects': 0,
                'movement_trend': 'initial'
            }

        prev_detections = self.prev_analysis.get('detections', [])
        prev_classes = Counter(d['class_name'] for d in prev_detections)
        curr_classes = Counter(d['class_name'] for d in detections)

        # 计算新增和消失的目标
        new_objects = 0
        disappeared = 0

        for class_name in set(list(prev_classes.keys()) + list(curr_classes.keys())):
            diff = curr_classes.get(class_name, 0) - prev_classes.get(class_name, 0)
            if diff > 0:
                new_objects += diff
            elif diff < 0:
                disappeared += abs(diff)

        # 判断运动趋势
        total_change = new_objects + disappeared
        if total_change == 0:
            trend = 'stable'
        elif new_objects > disappeared:
            trend = 'increasing'
        elif disappeared > new_objects:
            trend = 'decreasing'
        else:
            trend = 'changing'

        return {
            'new_objects': new_objects,
            'disappeared_objects': disappeared,
            'movement_trend': trend,
            'change_magnitude': total_change
        }

    def _determine_scene_type(self, detections: List[Dict],
                               distribution: Dict) -> str:
        """
        确定场景类型
        基于目标数量、类型和分布特征
        """
        total_count = len(detections)

        # 空场景
        if total_count == 0:
            return 'empty'

        # 获取类别分布
        category_dist = distribution.get('by_category', {})
        vehicle_count = category_dist.get('vehicle', {}).get('count', 0)
        pedestrian_count = category_dist.get('pedestrian', {}).get('count', 0)

        # 危险场景：大量车辆或行人
        if vehicle_count >= 3 or pedestrian_count >= 5:
            return 'dangerous'

        # 拥挤场景：目标总数较多
        if total_count >= 6:
            return 'crowded'

        # 复杂场景：多种类型目标
        if len(category_dist) >= 3:
            return 'complex'

        # 简单场景
        return 'simple'

    def _get_category(self, class_name: str) -> str:
        """获取目标类别所属的分组"""
        for category, classes in self.category_groups.items():
            if class_name in classes:
                return category
        return 'other'

    def _get_dominant_side(self, left: int, center: int, right: int) -> str:
        """确定主要分布区域"""
        counts = {'left': left, 'center': center, 'right': right}
        return max(counts, key=counts.get)

    def _generate_summary(self, scene_type: str, distribution: Dict,
                          spatial: Dict, movement: Dict) -> str:
        """
        生成场景摘要
        用自然语言描述当前场景特征
        """
        # 场景类型描述
        type_desc = self.SCENE_TYPES.get(scene_type, '未知场景')

        # 主要目标描述
        by_class = distribution.get('by_class', {})
        main_objects = []
        for class_name, info in list(by_class.items())[:3]:
            count = info['count']
            class_names_cn = {
                'person': '行人',
                'car': '汽车',
                'truck': '卡车',
                'bus': '公交车',
                'bicycle': '自行车',
                'motorcycle': '摩托车',
                'dog': '狗',
                'traffic light': '红绿灯'
            }
            cn_name = class_names_cn.get(class_name, class_name)
            main_objects.append(f"{count}个{cn_name}")

        # 位置描述
        dominant_side = spatial.get('dominant_side', 'center')
        side_desc = {
            'left': '左侧',
            'center': '前方',
            'right': '右侧'
        }.get(dominant_side, '前方')

        # 组装摘要
        summary_parts = [type_desc]

        if main_objects:
            summary_parts.append(f"主要目标: {', '.join(main_objects)}")

        summary_parts.append(f"目标主要分布在{side_desc}")

        # 运动趋势描述
        trend = movement.get('movement_trend', 'stable')
        if trend == 'increasing':
            summary_parts.append("有新目标出现")
        elif trend == 'decreasing':
            summary_parts.append("有目标离开视野")

        return '，'.join(summary_parts)

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def reset(self):
        """重置分析器状态"""
        self.prev_analysis = None
        logger.info("场景分析器已重置")
