# -*- coding: utf-8 -*-
"""
BlindGuard 核心模块
包含系统的核心功能实现
"""

from .detection_engine import DetectionEngine
from .risk_evaluator import RiskEvaluator
from .voice_announcer import VoiceAnnouncer
from .scene_analyzer import SceneAnalyzer
from .config_manager import ConfigManager

__all__ = [
    'DetectionEngine',
    'RiskEvaluator',
    'VoiceAnnouncer',
    'SceneAnalyzer',
    'ConfigManager'
]

__version__ = '1.0.0'
__author__ = 'BlindGuard Team'
