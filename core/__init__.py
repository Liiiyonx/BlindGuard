# -*- coding: utf-8 -*-
"""
BlindGuard 核心模块
检测引擎 → 红绿灯状态/目标跟踪 → 风险评估 → 场景分析 → 智能体 → 语音播报

这里使用惰性导出，避免只导入一个轻量工具模块时把 YOLO/PyTorch 全部拉起。
`from core import DetectionPipeline` 等既有用法保持兼容。
"""

from .version import VERSION

__all__ = [
    'DetectionEngine',
    'RiskEvaluator',
    'VoiceAnnouncer',
    'SceneAnalyzer',
    'ConfigManager',
    'BlindGuardAgent',
    'load_agent_config',
    'SimpleTracker',
    'TrafficLightClassifier',
    'UserProfile',
    'DetectionPipeline',
    'VERSION',
]

_EXPORTS = {
    'DetectionEngine': ('.detection_engine', 'DetectionEngine'),
    'RiskEvaluator': ('.risk_evaluator', 'RiskEvaluator'),
    'VoiceAnnouncer': ('.voice_announcer', 'VoiceAnnouncer'),
    'SceneAnalyzer': ('.scene_analyzer', 'SceneAnalyzer'),
    'ConfigManager': ('.config_manager', 'ConfigManager'),
    'BlindGuardAgent': ('.agent', 'BlindGuardAgent'),
    'load_agent_config': ('.agent', 'load_agent_config'),
    'SimpleTracker': ('.tracker', 'SimpleTracker'),
    'TrafficLightClassifier': ('.traffic_light', 'TrafficLightClassifier'),
    'UserProfile': ('.user_profile', 'UserProfile'),
    'DetectionPipeline': ('.pipeline', 'DetectionPipeline'),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__version__ = VERSION
__author__ = 'BlindGuard Team'
