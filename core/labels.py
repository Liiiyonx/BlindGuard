# -*- coding: utf-8 -*-
"""
统一的类别中英文映射

此前中文名映射散落在 config.yaml、agent.py、risk_evaluator.py、scene_analyzer.py
四处，容易互相不一致。本模块是唯一默认来源；
config.yaml 的 class_mapping 在应用装配时传入各模块作为覆盖项。
"""

CLASS_NAME_CN = {
    # BlindGuard 自定义训练类别（best.pt，下划线命名）
    'manhole_cover': '井盖',
    'public_facility': '公共设施',
    'tactile_paving': '盲道',
    'traffic_light': '红绿灯',

    # 交通参与者
    'person': '行人',
    'bicycle': '自行车',
    'car': '汽车',
    'motorcycle': '摩托车',
    'bus': '公交车',
    'truck': '卡车',
    'train': '火车',
    'airplane': '飞机',
    'boat': '船',

    # 交通设施
    'traffic light': '红绿灯',
    'stop sign': '停车标志',
    'fire hydrant': '消防栓',
    'parking meter': '停车计时器',
    'bench': '长椅',
    'pole': '电线杆',

    # 动物
    'dog': '狗',
    'cat': '猫',
    'bird': '鸟',
    'horse': '马',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑马',
    'giraffe': '长颈鹿',

    # 常见障碍物 / 物品
    'backpack': '背包',
    'umbrella': '雨伞',
    'handbag': '手提包',
    'suitcase': '行李箱',
    'chair': '椅子',
    'couch': '沙发',
    'potted plant': '盆栽',
}


def get_class_cn(class_name, mapping=None):
    """查询类别中文名：先查覆盖映射，再查默认表，查不到原样返回"""
    if not class_name:
        return '目标'
    if mapping and class_name in mapping:
        return mapping[class_name]
    return CLASS_NAME_CN.get(class_name, class_name)
