# -*- coding: utf-8 -*-
"""
配置管理模块
负责系统配置的加载、保存和管理

本模块实现了配置文件的读写功能，
支持JSON/YAML格式配置和环境变量覆盖，
为系统各组件提供统一的配置访问接口。
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# 尝试导入YAML支持
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger('BlindGuard.Config')


class ConfigManager:
    """
    配置管理器
    管理系统配置的生命周期
    """

    # 默认配置
    DEFAULT_CONFIG = {
        # 模型配置
        'model': {
            'path': 'yolov8n.pt',
            'confidence_threshold': 0.45,
            'iou_threshold': 0.5,
            'device': ''
        },

        # 风险评估配置
        'risk': {
            'thresholds': {
                'class_scores': {},
                'distance': {
                    'very_close': 0.20,
                    'close': 0.10,
                    'medium': 0.04,
                    'far': 0.01
                }
            }
        },

        # 语音配置
        'voice': {
            'rate': 160,
            'volume': 1.0,
            'cooldown': 3.0,
            'language': 'chinese'
        },

        # 摄像头配置
        'camera': {
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 15
        },

        # 系统配置
        'system': {
            'log_level': 'INFO',
            'log_dir': 'logs',
            'upload_dir': 'uploads',
            'max_log_size': 10485760,  # 10MB
            'backup_count': 5
        },

        # 界面配置
        'ui': {
            'theme': 'dark',
            'language': 'zh-CN',
            'show_fps': True,
            'show_detection_count': True
        }
    }

    # 环境变量前缀
    ENV_PREFIX = 'BLINDGUARD_'

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，None则自动查找配置文件
        """
        # 配置文件路径
        if config_path:
            self.config_path = Path(config_path)
        else:
            # 自动查找配置文件（优先YAML）
            self.config_path = self._find_config_file()

        # 运行时配置
        self._config = {}

        # 加载配置
        self._load_config()

    def _find_config_file(self) -> Path:
        """自动查找配置文件"""
        # 优先查找YAML格式
        yaml_paths = [
            Path('config.yaml'),
            Path('config.yml'),
            Path('config/config.yaml'),
        ]
        for path in yaml_paths:
            if path.exists():
                return path

        # 其次查找JSON格式
        json_paths = [
            Path('config.json'),
            Path('config/config.json'),
        ]
        for path in json_paths:
            if path.exists():
                return path

        # 默认使用YAML
        return Path('config.yaml')

    def _load_config(self):
        """加载配置"""
        # 从默认配置开始
        self._config = self._deep_copy(self.DEFAULT_CONFIG)

        # 尝试从文件加载
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    # 根据文件扩展名选择解析器
                    if self.config_path.suffix in ('.yaml', '.yml'):
                        if HAS_YAML:
                            file_config = yaml.safe_load(f)
                        else:
                            logger.warning("YAML配置文件需要安装pyyaml: pip install pyyaml")
                            file_config = {}
                    else:
                        file_config = json.load(f)

                    if file_config:
                        self._convert_config(file_config)
                        self._merge_config(self._config, file_config)
                        logger.info(f"已加载配置文件: {self.config_path}")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")

        # 从环境变量加载
        self._load_from_env()

        logger.info("配置加载完成")

    def _convert_config(self, config: Dict):
        """转换原始配置格式到新格式"""
        # 转换摄像头配置
        if 'camera' in config:
            camera = config['camera']
            if 'index' in camera:
                camera['device_id'] = camera.pop('index')

        # 转换检测器配置
        if 'detector' in config:
            detector = config['detector']
            config['model'] = {
                'path': detector.get('model_path', 'best.pt'),
                'confidence_threshold': detector.get('confidence', 0.5),
                'iou_threshold': detector.get('iou_threshold', 0.45),
                'device': detector.get('device', ''),
                'img_size': detector.get('img_size', 640)
            }

        # 转换风险引擎配置
        if 'risk_engine' in config:
            risk_engine = config['risk_engine']
            config['risk'] = {
                'thresholds': risk_engine.get('area_thresholds', {}),
                'high_risk_classes': risk_engine.get('high_risk_classes', []),
                'medium_risk_classes': risk_engine.get('medium_risk_classes', []),
                'low_risk_classes': risk_engine.get('low_risk_classes', [])
            }

        # 转换语音配置
        if 'announcer' in config:
            announcer = config['announcer']
            config['voice'] = {
                'rate': announcer.get('rate', 180),
                'volume': announcer.get('volume', 1.0),
                'cooldown': announcer.get('cooldown', 3.0)
            }

        # 保留类别映射
        if 'class_mapping' in config:
            config['class_mapping'] = config['class_mapping']

    def _load_from_env(self):
        """从环境变量加载配置"""
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # 转换环境变量名到配置路径
                config_key = key[len(self.ENV_PREFIX):].lower().replace('_', '.')
                self._set_nested(config_key, self._parse_env_value(value))

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试解析为布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 返回字符串
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的路径 (如 'model.path')
            default: 默认值

        Returns:
            配置值
        """
        return self._get_nested(key, default)

    def set(self, key: str, value: Any, save: bool = False):
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
            save: 是否立即保存到文件
        """
        self._set_nested(key, value)

        if save:
            self.save()

        logger.debug(f"配置已更新: {key} = {value}")

    def save(self):
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已保存到: {self.config_path}")

        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def get_section(self, section: str) -> Dict:
        """
        获取配置段

        Args:
            section: 段名称

        Returns:
            配置段字典
        """
        return self._config.get(section, {})

    def update_section(self, section: str, values: Dict):
        """
        更新配置段

        Args:
            section: 段名称
            values: 新的配置值
        """
        if section not in self._config:
            self._config[section] = {}

        self._merge_config(self._config[section], values)

    def _get_nested(self, key: str, default: Any = None) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def _set_nested(self, key: str, value: Any):
        """设置嵌套配置值"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _merge_config(self, base: Dict, override: Dict):
        """深度合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _deep_copy(self, obj: Any) -> Any:
        """深拷贝对象"""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def reset_to_default(self):
        """重置为默认配置"""
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        logger.info("配置已重置为默认值")

    def export_config(self) -> Dict:
        """导出当前配置"""
        return self._deep_copy(self._config)

    def import_config(self, config: Dict):
        """导入配置"""
        self._config = self._deep_copy(config)
        logger.info("配置已导入")

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            配置是否有效
        """
        try:
            # 检查必要字段
            required_sections = ['model', 'risk', 'voice', 'camera', 'system']
            for section in required_sections:
                if section not in self._config:
                    logger.error(f"缺少必要配置段: {section}")
                    return False

            # 检查数值范围
            voice = self._config.get('voice', {})
            if not (50 <= voice.get('rate', 160) <= 300):
                logger.error("语音速率超出范围")
                return False

            if not (0.0 <= voice.get('volume', 1.0) <= 1.0):
                logger.error("音量超出范围")
                return False

            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
