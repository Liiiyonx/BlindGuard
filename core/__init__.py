# -*- coding: utf-8 -*-
"""
BlindGuard 核心模块
检测引擎 → 风险评估 → 场景分析 → 智能体 → 语音播报
"""

from .detection_engine import DetectionEngine
from .risk_evaluator import RiskEvaluator
from .voice_announcer import VoiceAnnouncer
from .scene_analyzer import SceneAnalyzer
from .config_manager import ConfigManager
from .agent import BlindGuardAgent, load_agent_config

__all__ = [
    'DetectionEngine',
    'RiskEvaluator',
    'VoiceAnnouncer',
    'SceneAnalyzer',
    'ConfigManager',
    'BlindGuardAgent',
    'load_agent_config',
]

__version__ = '2.0.0'
__author__ = 'BlindGuard Team'
