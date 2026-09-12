# -*- coding: utf-8 -*-
"""
BlindGuard 核心模块
检测引擎 → 红绿灯状态/目标跟踪 → 风险评估 → 场景分析 → 智能体 → 语音播报
"""

from .detection_engine import DetectionEngine
from .risk_evaluator import RiskEvaluator
from .voice_announcer import VoiceAnnouncer
from .scene_analyzer import SceneAnalyzer
from .config_manager import ConfigManager
from .agent import BlindGuardAgent, load_agent_config
from .tracker import SimpleTracker
from .traffic_light import TrafficLightClassifier
from .user_profile import UserProfile

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
]

__version__ = '2.1.0'
__author__ = 'BlindGuard Team'
