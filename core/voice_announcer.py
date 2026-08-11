# -*- coding: utf-8 -*-
"""
语音播报模块
提供文本转语音(TTS)播报功能

本模块实现了智能语音播报系统，
支持优先级队列、冷却机制和异步播报，
确保关键风险信息能够及时传达给用户。
"""

import time
import logging
import threading
from typing import Optional
from queue import PriorityQueue, Empty

logger = logging.getLogger('BlindGuard.Voice')


class VoiceAnnouncer:
    """
    语音播报器
    管理语音播报队列，实现智能播报策略
    """

    # 优先级定义（数值越小优先级越高）
    PRIORITY_CRITICAL = 0  # 极高风险 - 立即播报
    PRIORITY_HIGH = 1      # 高风险 - 尽快播报
    PRIORITY_MEDIUM = 2    # 中等风险 - 正常播报
    PRIORITY_LOW = 3       # 低风险 - 延迟播报
    PRIORITY_SYSTEM = 4    # 系统消息 - 正常播报

    def __init__(self, rate: int = 160, volume: float = 1.0,
                 cooldown: float = 3.0, language: str = 'chinese'):
        """
        初始化语音播报器

        Args:
            rate: 语速 (字/分钟)
            volume: 音量 (0.0-1.0)
            cooldown: 相同消息冷却时间 (秒)
            language: 语言设置
        """
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown
        self.language = language

        # TTS引擎
        self.engine = None
        self.engine_lock = threading.Lock()

        # 播报队列
        self.queue = PriorityQueue()
        self.is_running = False
        self.worker_thread = None

        # 消息历史（用于去重和冷却）
        self.message_history = {}
        self.history_lock = threading.Lock()

        # 统计信息
        self.total_spoken = 0
        self.total_skipped = 0

        # 初始化引擎
        self._init_engine()

    def _init_engine(self):
        """初始化TTS引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()

            # 设置语速
            self.engine.setProperty('rate', self.rate)

            # 设置音量
            self.engine.setProperty('volume', self.volume)

            # 尝试设置中文语音
            self._set_chinese_voice()

            logger.info(f"TTS引擎初始化成功 (语速:{self.rate}, 音量:{self.volume})")

        except ImportError:
            logger.warning("pyttsx3未安装，语音功能不可用")
            self.engine = None
        except Exception as e:
            logger.error(f"TTS引擎初始化失败: {e}")
            self.engine = None

    def _set_chinese_voice(self):
        """设置中文语音"""
        if self.engine is None:
            return

        try:
            voices = self.engine.getProperty('voices')

            # 优先查找中文语音
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()

                if 'chinese' in voice_name or 'zh' in voice_id:
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"已设置中文语音: {voice.name}")
                    return

            # 如果没有找到中文语音，使用默认语音
            logger.warning("未找到中文语音，使用默认语音")

        except Exception as e:
            logger.warning(f"设置语音失败: {e}")

    def start(self):
        """启动播报服务"""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name='VoiceWorker'
        )
        self.worker_thread.start()
        logger.info("语音播报服务已启动")

    def stop(self):
        """停止播报服务"""
        self.is_running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=3)
            self.worker_thread = None

        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass

        logger.info("语音播报服务已停止")

    def speak(self, text: str, priority: int = PRIORITY_MEDIUM,
              force: bool = False) -> bool:
        """
        添加播报任务

        Args:
            text: 要播报的文本
            priority: 优先级 (0-4)
            force: 是否强制播报（忽略冷却）

        Returns:
            是否成功添加到队列
        """
        if not text or not text.strip():
            return False

        if self.engine is None:
            logger.warning("TTS引擎不可用")
            return False

        # 检查冷却
        if not force and self._is_in_cooldown(text):
            self.total_skipped += 1
            return False

        # 添加到队列
        try:
            self.queue.put((priority, time.time(), text), block=False)
            return True
        except Exception as e:
            logger.error(f"添加播报任务失败: {e}")
            return False

    def _is_in_cooldown(self, text: str) -> bool:
        """检查消息是否在冷却期内"""
        with self.history_lock:
            current_time = time.time()
            last_time = self.message_history.get(text, 0)
            return (current_time - last_time) < self.cooldown

    def _process_queue(self):
        """处理播报队列的工作线程"""
        while self.is_running:
            try:
                # 从队列获取任务，超时1秒
                priority, timestamp, text = self.queue.get(timeout=1.0)

                # 执行播报
                self._do_speak(text)

                # 更新消息历史
                with self.history_lock:
                    self.message_history[text] = time.time()

                # 清理过期的历史记录
                self._cleanup_history()

                self.total_spoken += 1

            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"处理播报队列异常: {e}")
                time.sleep(0.1)

    def _do_speak(self, text: str):
        """执行实际的语音播报"""
        with self.engine_lock:
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError as e:
                if "run loop already started" in str(e):
                    logger.warning(f"语音引擎冲突，重试: {e}")
                    time.sleep(0.1)
                    try:
                        self.engine.stop()
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as retry_error:
                        logger.error(f"重试播报失败: {retry_error}")
                else:
                    logger.error(f"播报运行时错误: {e}")
            except Exception as e:
                logger.error(f"播报执行失败: {e}")

    def _cleanup_history(self):
        """清理过期的历史记录"""
        with self.history_lock:
            current_time = time.time()
            expired_keys = [
                key for key, last_time in self.message_history.items()
                if current_time - last_time > self.cooldown * 10
            ]
            for key in expired_keys:
                del self.message_history[key]

    def set_rate(self, rate: int):
        """设置语速"""
        self.rate = max(50, min(300, rate))
        if self.engine:
            with self.engine_lock:
                self.engine.setProperty('rate', self.rate)
        logger.info(f"语速已设置为: {self.rate}")

    def set_volume(self, volume: float):
        """设置音量"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            with self.engine_lock:
                self.engine.setProperty('volume', self.volume)
        logger.info(f"音量已设置为: {self.volume}")

    def set_cooldown(self, seconds: float):
        """设置冷却时间"""
        self.cooldown = max(0.5, min(30.0, seconds))
        logger.info(f"冷却时间已设置为: {self.cooldown}秒")

    def clear_queue(self):
        """清空播报队列"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        logger.info("播报队列已清空")

    def get_stats(self) -> dict:
        """获取播报统计信息"""
        return {
            'is_running': self.is_running,
            'queue_size': self.queue.qsize(),
            'total_spoken': self.total_spoken,
            'total_skipped': self.total_skipped,
            'rate': self.rate,
            'volume': self.volume,
            'cooldown': self.cooldown
        }

    def speak_critical(self, text: str) -> bool:
        """播报极高风险消息"""
        return self.speak(text, priority=self.PRIORITY_CRITICAL, force=True)

    def speak_high(self, text: str) -> bool:
        """播报高风险消息"""
        return self.speak(text, priority=self.PRIORITY_HIGH)

    def speak_system(self, text: str) -> bool:
        """播报系统消息"""
        return self.speak(text, priority=self.PRIORITY_SYSTEM, force=True)
