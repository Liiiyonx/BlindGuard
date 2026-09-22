# -*- coding: utf-8 -*-
"""
用户偏好长期记忆（JSON 持久化）

Agent 可通过 set_user_preference 工具在对话中调整用户偏好
（如"说慢一点" -> voice_rate），应用层负责把偏好真正应用到设备上。
数据保存在 data/user_profile.json，跨重启生效。
"""

import json
import logging
import os
import threading
from typing import Dict, Tuple

logger = logging.getLogger('BlindGuard.Profile')


class UserProfile:
    """持久化的用户偏好存储（线程安全）"""

    # 数值型偏好：key -> (最小值, 最大值)
    LIMITS = {
        'voice_rate': (80, 300),
    }
    # 枚举型偏好：key -> 合法取值
    ENUMS = {
        'announce_detail': ('concise', 'detailed'),
    }
    CN_NAMES = {
        'voice_rate': '语速',
        'announce_detail': '播报详略',
    }

    def __init__(self, path: str = 'data/user_profile.json'):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f) or {}
        except Exception as e:
            logger.warning(f"读取用户偏好失败（将使用空偏好）: {e}")
            self._data = {}

    def _save_locked(self):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_all(self) -> Dict:
        with self._lock:
            return dict(self._data)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value) -> Tuple[bool, str]:
        """
        设置偏好，带类型/范围校验

        Returns:
            (是否成功, 面向用户的提示消息)
        """
        key = str(key or '').strip()
        if not key:
            return False, '偏好项名称不能为空'
        if key not in self.LIMITS and key not in self.ENUMS:
            return False, f"不支持的偏好项: {key}"

        try:
            if key in self.LIMITS:
                value = int(float(value))
                lo, hi = self.LIMITS[key]
                if not (lo <= value <= hi):
                    return False, f"{self.CN_NAMES.get(key, key)}需在 {lo}-{hi} 之间"
            elif key in self.ENUMS:
                value = str(value).strip()
                if value not in self.ENUMS[key]:
                    legal = ' 或 '.join(self.ENUMS[key])
                    return False, f"{self.CN_NAMES.get(key, key)}仅支持 {legal}"
            else:
                value = str(value)
        except (TypeError, ValueError):
            return False, f"{self.CN_NAMES.get(key, key)}的值无效: {value}"

        with self._lock:
            self._data[key] = value
            try:
                self._save_locked()
            except Exception as e:
                logger.error(f"保存用户偏好失败: {e}")
                return False, '偏好保存失败（写入磁盘出错）'
        return True, f"已将{self.CN_NAMES.get(key, key)}设置为 {value}"
