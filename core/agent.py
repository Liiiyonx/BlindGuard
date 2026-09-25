# -*- coding: utf-8 -*-
"""
智能体核心模块（V2.5）

在传统 CV 管线（检测→跟踪→风险→语音）之上叠加编排式多能力 Agent：
1. 安全守门员：红灯/黄灯/高风险/未知灯色确定性拦截，任何输出都不得授权通行
2. 工具注册表：统一 schema 校验、读写权限、安全级别、超时、异常和调用审计
3. 专业能力：实时感知、风险分析、时序事件、视觉观察、偏好管家、运行监控
4. 编排循环：本地意图路由 → 行动 → 安全复核 → 有界运行轨迹；
   规则未命中时由 LLM 角色规划器做任务分解（只决定参与角色，工具由角色表推导）
5. 主动播报：紧急度评分、原因排序、LLM/模板生成、安全复核和历史上限
6. 离线兜底：未配置密钥或调用失败时回退模板播报与本地确定性对话

LLM 走 OpenAI 兼容 /chat/completions，仅用标准库 urllib，不引入新依赖。
"""

import os
import json
import time
import base64
import logging
import threading
import urllib.request
import urllib.error
from collections import deque
from typing import Dict, List, Optional

from core.agent_orchestrator import (
    AgentOrchestrator,
    SPECIALISTS,
    assess_announcement,
)
from core.agent_tools import ToolRegistry
from core.config_manager import ConfigManager
from core.geometry import position_zone
from core.labels import get_class_cn
from core import safety_policy
from core.traffic_light import STATE_CN

logger = logging.getLogger('BlindGuard.Agent')

# 风险等级排序，数值越小越紧急
RISK_ORDER = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'safe': 4}

TREND_CN = {'approaching': '正在接近', 'receding': '正在远离'}

ZONE_CN = {'left': '左侧', 'center': '正前方', 'right': '右侧'}

SYSTEM_PROMPT = (
    "你是 BlindGuard 智能导盲系统的核心助手，服务对象是视障人士。"
    "你会收到摄像头检测出的目标（类别、风险等级、方位、红绿灯状态、运动趋势）以及最近若干帧的时序记忆。"
    "安全规则具有最高优先级：你只能描述检测结果，绝不能批准、建议或保证用户通行。"
    "不得输出“可以通行”“安全通行”“放心走”“能过马路”等肯定结论。"
    "即使看到绿灯，也必须说明系统无法确认路口是否安全，请用户结合盲杖、交通规则和人工判断。"
    "任务一【播报】：生成一句简短、自然、口语化的中文语音提示，"
    "优先提醒高风险、近距离、正前方、正在接近的目标；不超过 20 个字，不要堆砌细节，不要解释自己是 AI。"
    "任务二【对话】：回答用户关于周围环境的提问，回答不超过 80 字，基于检测结果与工具结果如实描述，"
    "看不到的信息不要编造。始终使用中文。"
)

SYSTEM_PROMPT_WITH_TOOLS = SYSTEM_PROMPT + (
    "你还可以调用工具获取更完整的信息：当问题超出当前上下文"
    "（如需要确认画面细节、历史事件、用户偏好）时应主动调用工具，不要凭空猜测。"
)

# 注册给 LLM 的工具集（OpenAI function calling 格式）
AGENT_TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_current_detections',
        'description': '获取当前画面检测到的全部目标列表（名称、风险等级、方位、置信度、'
                       '红绿灯状态、运动趋势、已持续秒数）',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_scene_overview',
        'description': '获取场景级信息：整体风险等级、场景类型、各类目标数量、运动趋势',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_memory_summary',
        'description': '获取最近若干帧的时序记忆（目标出现/消失/接近/远离的历史）',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'look_at_frame',
        'description': '调用视觉语言模型直接查看当前画面。用于识别文字/标志/红绿灯读秒等'
                       'YOLO 检测框之外的信息，或进一步确认画面细节。调用有数秒延迟，仅在必要时使用',
        'parameters': {'type': 'object', 'properties': {
            'question': {'type': 'string', 'description': '想确认的具体问题，如"现在亮的是红灯吗"'},
        }, 'required': ['question']},
    }},
    {'type': 'function', 'function': {
        'name': 'get_user_preferences',
        'description': '读取用户偏好设置（语速、播报详略等）',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'set_user_preference',
        'description': '设置用户偏好。当用户明确提出调整要求时调用，如"说慢一点"→voice_rate',
        'parameters': {'type': 'object', 'properties': {
            'key': {'type': 'string', 'enum': ['voice_rate', 'announce_detail'],
                    'description': '偏好项'},
            'value': {'type': 'string',
                      'description': '偏好值：voice_rate 为 80-300 的数字；'
                                     'announce_detail 为 concise(简短) 或 detailed(详细)'},
        }, 'required': ['key', 'value']},
    }},
    {'type': 'function', 'function': {
        'name': 'get_risk_assessment',
        'description': '获取结构化风险分析：整体风险、优先避让目标、风险原因、'
                       '各等级数量与安全边界说明',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_navigation_guidance',
        'description': '获取非通行类无障碍导航建议：停步、减速、盲杖确认、'
                       '左右侧障碍分布和优先避让对象。该工具不会授权过街或通行',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_environment_trends',
        'description': '获取近期事件流和运动趋势，包括目标出现/消失、接近/远离、'
                       '红绿灯变化和风险升级；用于回答“刚才发生了什么”类问题',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_system_health',
        'description': '获取智能体运行健康：LLM 在线/降级、工具协议、VLM、'
                       '记忆帧数、最近工具调用与错误信息',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_agent_capabilities',
        'description': '获取智能体架构、专业角色、工具清单及其读写权限，'
                       '用于解释系统能做什么和安全边界',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_announcement_history',
        'description': '获取最近主动语音播报及其风险、紧急度和生成路径',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
    {'type': 'function', 'function': {
        'name': 'get_alert_policy',
        'description': '解释当前是否达到主动播报门槛、紧急度、冷却和保守静默原因；'
                       '静默只表示未达到播报门槛，不代表环境安全',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    }},
]

# 兼容旧调用方的默认值；实例可通过 agent.architecture.max_tool_rounds 覆盖
MAX_TOOL_ROUNDS = 3


class BlindGuardAgent:
    """智能导盲 Agent：LLM 推理 + 工具调用 + 时序记忆 + 多模态看图"""

    def __init__(self, config: Optional[Dict] = None, class_mapping: Optional[Dict] = None):
        config = config or {}
        self.enabled = bool(config.get('enabled', True))
        self._class_mapping = dict(class_mapping) if class_mapping else {}

        llm = config.get('llm', {}) or {}
        self.base_url = (llm.get('base_url') or '').rstrip('/')
        self.api_key = llm.get('api_key') or ''
        self.model = llm.get('model') or 'deepseek-chat'
        self.temperature = float(llm.get('temperature', 0.6))
        self.timeout = float(llm.get('timeout', 8))

        # 多模态看图（可选）：独立的 OpenAI 兼容视觉端点
        vis = config.get('vision', {}) or {}
        self.vision_base_url = (vis.get('base_url') or '').rstrip('/')
        self.vision_api_key = vis.get('api_key') or ''
        self.vision_model = vis.get('model') or ''
        self.vlm_timeout = float(vis.get('timeout', 20))
        self.vlm_ok = bool(self.enabled and self.vision_base_url
                           and self.vision_api_key and self.vision_model)

        ann = config.get('announce', {}) or {}
        self.announce_cooldown = float(ann.get('cooldown', 3.0))
        self.memory_size = int(ann.get('memory_size', 8))
        # 主动播报的保守策略（分级告警）：低于该等级/置信度的目标不主动播报
        self.min_level = str(ann.get('min_level', 'medium'))
        self.min_confidence = float(ann.get('min_confidence', 0.45))

        chat = config.get('chat', {}) or {}
        self.max_history = int(chat.get('max_history', 6))

        architecture = config.get('architecture', {}) or {}
        self.agent_version = str(architecture.get('version', '2.5.0'))
        self.max_tool_rounds = int(
            architecture.get('max_tool_rounds', MAX_TOOL_ROUNDS)
        )
        self.max_model_tool_calls = int(
            architecture.get('max_model_tool_calls', 6)
        )
        self.max_specialist_roles = int(
            architecture.get('max_specialist_roles', 3)
        )
        self.tool_trace_size = int(architecture.get('tool_trace_size', 80))
        self.event_history_size = int(
            architecture.get('event_history_size', 40)
        )
        self.announcement_history_size = int(
            architecture.get('announcement_history_size', 20)
        )
        self.intent_planner_enabled = bool(
            architecture.get('enable_intent_planner', True)
        )
        self.tool_prefetch_enabled = bool(
            architecture.get('enable_tool_prefetch', True)
        )
        self.specialist_execution_enabled = bool(
            architecture.get('enable_specialist_team', True)
        )
        # 规则路由未命中时是否让 LLM 参与任务分解（只决定角色，不决定工具）
        self.llm_planner_enabled = bool(
            architecture.get('enable_llm_planner', True)
        )
        self.llm_planner_timeout = float(
            architecture.get('llm_planner_timeout', 8.0)
        )

        # 端点是否支持 tools（首次报错自动降级为纯上下文模式）
        self.tools_supported = True

        # LLM 熔断：连续失败达到阈值后暂停调用一段时间，
        # 避免端点不可达时每次对话/播报都傻等超时
        self._fail_count = 0
        self._fail_skip_until = 0.0
        self._llm_degraded = False
        self._last_error = ''
        self.FAIL_TRIP_THRESHOLD = 3
        self.FAIL_SKIP_SECONDS = 30.0

        # 应用上下文（由 BlindGuardApp 注入，提供工具所需的数据源）
        self.context = None

        # 共享状态锁：检测管线线程（update_memory）、播报后台线程
        # （generate_announcement）与 Flask Web 线程（chat/状态接口）会并发
        # 读写下方 5 个结构。用可重入锁保护，杜绝“遍历时被另一线程追加”
        # 引发的 RuntimeError: deque mutated during iteration。
        # 读路径一律在锁内取 list 快照、锁外迭代，避免长持锁阻塞检测管线。
        self._state_lock = threading.RLock()

        # 时序记忆：滑动窗口，每帧一个快照
        self.memory: deque = deque(maxlen=self.memory_size)
        # 从帧间差异聚合的事件流，避免把连续帧当成独立风险
        self.event_history: deque = deque(
            maxlen=max(1, self.event_history_size)
        )
        # 对话历史 [(role, content), ...]
        self.chat_history: List[Dict] = []

        # 最后一帧的缓存（供无 context 时兜底，如冒烟测试直接实例化 Agent）
        self._last_detections: List[Dict] = []
        self._last_risk = 'safe'
        self._last_frame_size = (640, 480)
        self._last_scene: Dict = {}

        self.last_announce_time = 0.0
        self.last_announce_text = ''
        self.announcement_history: deque = deque(
            maxlen=max(1, self.announcement_history_size)
        )

        # 工具注册表和编排层：外部仍只依赖 BlindGuardAgent 公共接口
        self.tool_registry = ToolRegistry()
        self._register_tools()
        self.orchestrator = AgentOrchestrator(
            self,
            max_tool_rounds=self.max_tool_rounds,
            trace_size=self.tool_trace_size,
            max_model_tool_calls=self.max_model_tool_calls,
            max_specialist_roles=self.max_specialist_roles,
            llm_planner_timeout=self.llm_planner_timeout,
        )
        self.orchestrator.intent_planner_enabled = (
            self.intent_planner_enabled
        )
        self.orchestrator.tool_prefetch_enabled = self.tool_prefetch_enabled
        self.orchestrator.specialist_execution_enabled = (
            self.specialist_execution_enabled
        )
        self.orchestrator.llm_planner_enabled = self.llm_planner_enabled

        # LLM 是否真正可用（需同时有 base_url 和 api_key）
        self.llm_ok = bool(self.enabled and self.base_url and self.api_key)
        if self.enabled and not self.llm_ok:
            logger.warning(
                "Agent 已启用但 LLM 未完整配置(base_url/api_key)，"
                "将回退到模板播报与本地对话"
            )

        logger.info(
            f"Agent 初始化: enabled={self.enabled} llm_ok={self.llm_ok} "
            f"tools={'auto' if self.tools_supported else 'off'} vlm_ok={self.vlm_ok} "
            f"model={self.model} memory_size={self.memory_size}"
        )

    # ==================== 上下文注入 ====================

    def set_context(self, context):
        """
        注入应用上下文（BlindGuardApp），需提供以下方法（ duck typing ）：
        - get_detections() -> List[Dict]   当前帧检测结果
        - get_overall_risk() -> str        整体风险等级
        - get_scene() -> Dict              场景信息（瘦身后）
        - get_frame_jpeg() -> bytes|None   当前画面 JPEG（供看图工具）
        - get_preferences() -> Dict        用户偏好
        - set_preference(key, value) -> Dict  设置偏好
        """
        self.context = context

    # ==================== 每帧状态更新 ====================

    def update_memory(self, detections: List[Dict], risk_level: str = 'safe',
                      frame_w: int = 640, frame_h: int = 480,
                      scene: Optional[Dict] = None):
        """每帧调用，更新滑动窗口记忆与内部缓存（不触发 LLM）"""
        self._last_detections = list(detections)
        self._last_risk = risk_level
        self._last_frame_size = (frame_w, frame_h)
        if scene is not None:
            self._last_scene = scene
        snapshot = self._summarize_frame(detections, risk_level, frame_w, frame_h)
        # 事件跟踪（读 memory[-1]、写 event_history）与记忆追加须与读路径互斥
        with self._state_lock:
            self._track_events(snapshot)
            self.memory.append(snapshot)

    def _summarize_frame(self, detections, risk_level, frame_w, frame_h) -> Dict:
        items = []
        denom = max(1, frame_w * frame_h)
        for d in detections:
            bbox = d.get('bbox') or [0, 0, 0, 0]
            if len(bbox) < 4:
                bbox = [0, 0, 0, 0]
            cx = (bbox[0] + bbox[2]) / 2
            w = max(0, bbox[2] - bbox[0])
            h = max(0, bbox[3] - bbox[1])
            item = {
                'name': self._cn(d),
                'risk': d.get('risk_level', 'low'),
                'zone': self._zone(cx, frame_w),
                'area': round((w * h) / denom, 4),
                'conf': round(float(d.get('confidence', 0)), 3),
            }
            if d.get('light_state'):
                item['light'] = d['light_state']
            if d.get('trend') in ('approaching', 'receding'):
                item['trend'] = d['trend']
            if d.get('track_id') is not None:
                item['id'] = d['track_id']
            items.append(item)
        return {'t': round(time.time(), 1), 'risk': risk_level, 'items': items}

    def _track_events(self, snapshot: Dict) -> None:
        """Aggregate frame-to-frame changes into a bounded semantic event log."""
        previous = self.memory[-1] if self.memory else None
        if previous is None:
            self._append_event(
                event_type="memory_started",
                severity=snapshot.get("risk", "safe"),
                text="开始记录环境事件",
                target="",
            )
            return

        old_items = {
            self._event_item_key(item): item
            for item in previous.get("items", [])
        }
        new_items = {
            self._event_item_key(item): item
            for item in snapshot.get("items", [])
        }

        for key, item in new_items.items():
            old = old_items.get(key)
            risk = str(item.get("risk") or "safe")
            name = str(item.get("name") or "目标")
            if old is None and risk in ("critical", "high", "medium"):
                self._append_event(
                    event_type="target_appeared",
                    severity=risk,
                    text=f"{name}进入视野",
                    target=key,
                )
            if item.get("trend") == "approaching" and (
                old is None or old.get("trend") != "approaching"
            ):
                self._append_event(
                    event_type="approaching",
                    severity=risk,
                    text=f"{name}正在接近",
                    target=key,
                )
            if item.get("trend") == "receding" and (
                old is None or old.get("trend") != "receding"
            ):
                self._append_event(
                    event_type="receding",
                    severity="low",
                    text=f"{name}正在远离",
                    target=key,
                )
            if item.get("light") and (
                old is None or old.get("light") != item.get("light")
            ):
                self._append_event(
                    event_type="traffic_light",
                    severity=risk,
                    text=(
                        f"红绿灯变为"
                        f"{STATE_CN.get(item['light'], item['light'])}"
                    ),
                    target=key,
                )

        for key, item in old_items.items():
            if key not in new_items and item.get("risk") in (
                "critical", "high", "medium"
            ):
                self._append_event(
                    event_type="target_left",
                    severity="low",
                    text=f"{item.get('name', '目标')}离开视野",
                    target=key,
                )

        old_rank = RISK_ORDER.get(previous.get("risk", "safe"), 4)
        new_rank = RISK_ORDER.get(snapshot.get("risk", "safe"), 4)
        if new_rank < old_rank:
            self._append_event(
                event_type="risk_escalated",
                severity=snapshot.get("risk", "safe"),
                text=(
                    f"整体风险从{previous.get('risk', 'safe')}"
                    f"升为{snapshot.get('risk', 'safe')}"
                ),
                target="overall",
            )

    def _append_event(self, *, event_type: str, severity: str,
                      text: str, target: str) -> None:
        now = time.time()
        # 事件去重窗口：在最近 N 条内按“类型+目标+时间”查重，而非只看最后一条，
        # 避免同一事件被交错的其他事件“隔开”后在短时间内重复记录
        with self._state_lock:
            for event in list(self.event_history)[-5:]:
                if (
                    event.get("type") == event_type
                    and event.get("target") == target
                    and now - float(event.get("timestamp", 0)) < 6.0
                ):
                    return
            self.event_history.append({
                "timestamp": round(now, 3),
                "time": time.strftime("%H:%M:%S"),
                "type": event_type,
                "severity": severity,
                "text": text,
                "target": target,
            })

    @staticmethod
    def _event_item_key(item: Dict) -> str:
        # 有 track_id（生产默认，跟踪器开启）时用它做实例级唯一键，无碰撞。
        # 无 track_id 的兜底用“类名:方位”粗粒度键：此时 trend 不会设置，
        # 接近/远离事件本就不触发，只剩 出现/离开/灯色 事件；同类同区的
        # 第二个目标被合并只会“少报”而非“误报”，方向保守、可接受。
        # 故意不按面积/位置细分——目标移动会让键跨帧漂移，导致同一目标被
        # 反复判为“离开+出现”，事件刷屏甚至触发多余播报（反保守，禁止）。
        if item.get("id") is not None:
            return f"id:{item.get('id')}"
        return f"{item.get('name', '')}:{item.get('zone', '')}"

    def _memory_text(self) -> str:
        # 锁内取 list 快照、锁外迭代，避免遍历 deque 时被检测管线线程追加
        with self._state_lock:
            memory_snapshot = list(self.memory)
        if not memory_snapshot:
            return '（暂无历史记忆）'
        lines = []
        n = len(memory_snapshot)
        for i, snap in enumerate(memory_snapshot):
            names = []
            for it in snap['items']:
                tag = ''
                if it.get('light'):
                    tag += f"[{STATE_CN.get(it['light'], it['light'])}]"
                if it.get('trend') == 'approaching':
                    tag += '[接近中]'
                elif it.get('trend') == 'receding':
                    tag += '[远离]'
                names.append(f"{it['name']}{tag}/{it['risk']}/{it['zone']}")
            lines.append(
                f"近第{n - i}帧[整体{snap['risk']}]: "
                + (', '.join(names) if names else '无目标')
            )
        event_lines = self._event_text(limit=4)
        if event_lines:
            lines.append("近期事件:\n" + event_lines)
        return '\n'.join(lines)

    def _event_text(self, limit: int = 8) -> str:
        # 锁内取快照，避免“判空”与“切片”之间被另一线程追加或清空
        with self._state_lock:
            events = list(self.event_history)
        if not events:
            return ""
        return "\n".join(
            f"[{event.get('time', '')}] {event.get('text', '')}"
            for event in events[-max(1, limit):]
        )

    # ==================== 播报生成 ====================

    def should_announce(self, detections: List[Dict]) -> List[Dict]:
        """
        主动播报的保守过滤（分级告警策略）：
        - 风险等级低于 min_level（如 low/safe）不主动播报，用户可主动查询
        - 置信度低于 min_confidence 的目标不播报——宁可漏报，不误报
        """
        min_order = RISK_ORDER.get(self.min_level, 2)
        result = []
        for d in detections:
            if RISK_ORDER.get(d.get('risk_level', 'safe'), 4) > min_order:
                continue
            if float(d.get('confidence', 0)) < self.min_confidence:
                continue
            result.append(d)
        return result

    def generate_announcement(self, detections: List[Dict],
                              risk_level: str = 'safe',
                              frame_w: int = 640,
                              frame_h: int = 480) -> Optional[str]:
        """
        生成一句播报文本。使用已积累的记忆，不再次更新记忆。

        无目标时返回 None（由调用方决定是否播报"前方通畅"）。
        """
        return self.orchestrator.generate_announcement(
            list(detections or []), risk_level, frame_w, frame_h
        )

    def _record_announcement(self, *, text: str, risk_level: str,
                             urgency: Dict, path: str) -> None:
        record = {
            'time': time.strftime('%H:%M:%S'),
            'timestamp': round(time.time(), 3),
            'text': text,
            'risk_level': risk_level,
            'urgency': urgency.get('urgency', 'low'),
            'score': urgency.get('score', 0),
            'reasons': list(urgency.get('reasons') or []),
            'path': path,
        }
        # 播报历史会被状态接口并发读，append 须在锁内
        with self._state_lock:
            self.announcement_history.append(record)

    def _fallback_announce(self, detections, frame_w: int = 640) -> str:
        """无 LLM 时的模板播报（含红绿灯与接近趋势）"""
        # 红灯优先：过马路场景的硬约束
        red = next((d for d in detections
                    if d.get('light_state') == 'red'), None)
        if red is not None:
            return "红灯，请等待，不要通行"
        yellow = next((d for d in detections
                       if d.get('light_state') == 'yellow'), None)
        if yellow is not None:
            return "黄灯，请等待，不要通行"

        best = min(
            detections,
            key=lambda d: (RISK_ORDER.get(d.get('risk_level', 'low'), 3),
                           -float(d.get('confidence', 0)))
        )
        cn = self._cn(best)
        level = best.get('risk_level', 'low')
        prefix = {'critical': '危险！', 'high': '危险！',
                  'medium': '注意，', 'low': '', 'safe': ''}.get(level, '')
        zone = self._zone(self._center_x(best), frame_w)
        if best.get('light_state') == 'green':
            return "绿灯，请自行确认路况，不能据此判断可通行"
        if best.get('trend') == 'approaching':
            return f"{prefix}{zone}{cn}正在接近，注意避让"
        return f"{prefix}{zone}有{cn}"

    def _llm_announce(self, detections, risk_level, frame_w, detail) -> str:
        limit = 40 if detail == 'detailed' else 20
        det_text = self._format_detections(detections, frame_w)
        user = (
            f"当前帧检测结果：\n{det_text}\n"
            f"最近记忆：\n{self._memory_text()}\n"
            f"整体风险：{risk_level}\n"
            f"请生成一句不超过 {limit} 字的中文语音播报。"
        )
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}]
        resp = self._call_llm(msgs, max_tokens=80 if detail == 'detailed' else 60)
        return self._clean(resp.get('content') or '')

    def _format_detections(self, detections, frame_w: int = 640) -> str:
        if not detections:
            return '无目标'
        lines = []
        for d in detections:
            conf = float(d.get('confidence', 0))
            zone = self._zone(self._center_x(d), frame_w)
            extra = ''
            if d.get('light_state'):
                extra += f", {STATE_CN.get(d['light_state'], d['light_state'])}"
            if d.get('trend') == 'approaching':
                extra += ', 正在接近'
            elif d.get('trend') == 'receding':
                extra += ', 正在远离'
            lines.append(
                f"- {self._cn(d)}: 风险{d.get('risk_level', 'low')}, "
                f"置信度{conf:.0%}, 方位{zone}{extra}"
            )
        return '\n'.join(lines)

    # ==================== 对话（工具调用循环） ====================

    def chat(self, user_message: str, detections: Optional[List[Dict]] = None,
             risk_level: str = 'safe', frame_w: int = 640,
             frame_h: int = 480) -> str:
        """
        用户主动提问。LLM 可在回答前调用工具（查检测/场景/记忆/看图/偏好），
        形成"感知-推理-行动"的 Agent 循环。
        """
        if detections is None:
            detections = self._current_detections()
        return self.orchestrator.chat(
            user_message,
            list(detections or []),
            risk_level,
            frame_w,
            frame_h,
        )

    def _remember_chat(self, user_message: str, reply: str):
        """Persist one clean user/assistant pair with bounded history."""
        # 对话历史会被 chat 与状态接口并发读写，整个追加/裁剪须在锁内
        with self._state_lock:
            self.chat_history.append({'role': 'user', 'content': user_message})
            self.chat_history.append({'role': 'assistant', 'content': reply})
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]

    def _fallback_chat(self, user_message: str, detections,
                       risk_level: str = 'safe') -> str:
        """无 LLM 时的本地对话：基于检测结果直接回答"""
        decision = safety_policy.evaluate_passage_question(
            user_message, detections, risk_level)
        if decision.is_passage_question:
            return decision.reply
        if not detections:
            if self._detection_health_warning():
                return '检测当前不可用，我无法确认周围情况。请停下脚步，用盲杖确认，不要凭本系统判断是否安全。'
            return '当前没有看到周围目标，请把摄像头对准环境后再问。'

        red = next((d for d in detections if d.get('light_state') == 'red'), None)
        yellow = next((d for d in detections if d.get('light_state') == 'yellow'), None)
        green = next((d for d in detections if d.get('light_state') == 'green'), None)
        light_hint = ''
        if red is not None:
            light_hint = '红灯亮，请等待，不要通行。'
        elif yellow is not None:
            light_hint = '黄灯亮，请等待，不要通行。'
        elif green is not None:
            light_hint = '检测到绿灯，但系统不能确认路口安全。'

        names = []
        for d in detections:
            tag = ''
            if d.get('trend') == 'approaching':
                tag = '(正在接近)'
            names.append(f"{self._cn(d)}({d.get('risk_level', 'low')}){tag}")
        worst = min(detections,
                    key=lambda d: RISK_ORDER.get(d.get('risk_level', 'low'), 3))
        return (f"我现在能看到：{'、'.join(names)}。"
                f"{self._cn(worst)}风险最高，请注意。{light_hint}")

    # ==================== 工具实现 ====================

    def _system_prompt(self, include_tools: bool = True) -> str:
        if include_tools and self.tools_supported:
            return SYSTEM_PROMPT_WITH_TOOLS
        return SYSTEM_PROMPT

    def _register_tools(self) -> None:
        """Register schemas and bounded handlers in one auditable registry."""
        handlers = {
            'get_current_detections': (
                self._tool_current_detections, 'read', 'standard', 2000,
            ),
            'get_scene_overview': (
                self._tool_scene_overview, 'read', 'standard', 2000,
            ),
            'get_memory_summary': (
                self._tool_memory_summary, 'read', 'standard', 2000,
            ),
            'look_at_frame': (
                self._tool_look_at_frame, 'read', 'external', 22000,
            ),
            'get_user_preferences': (
                self._tool_get_user_preferences, 'read', 'user_data', 2000,
            ),
            'set_user_preference': (
                self._tool_set_user_preference, 'write', 'user_data', 2000,
            ),
            'get_risk_assessment': (
                self._tool_risk_assessment, 'read', 'safety_context', 2000,
            ),
            'get_navigation_guidance': (
                self._tool_navigation_guidance,
                'read',
                'safety_context',
                2000,
            ),
            'get_environment_trends': (
                self._tool_environment_trends, 'read', 'standard', 2000,
            ),
            'get_system_health': (
                self._tool_system_health, 'read', 'system', 2000,
            ),
            'get_agent_capabilities': (
                self._tool_agent_capabilities, 'read', 'system', 2000,
            ),
            'get_announcement_history': (
                self._tool_announcement_history, 'read', 'user_data', 2000,
            ),
            'get_alert_policy': (
                self._tool_alert_policy, 'read', 'safety_context', 2000,
            ),
        }
        for definition in AGENT_TOOLS:
            name = (definition.get('function') or {}).get('name') or ''
            if name not in handlers:
                raise ValueError(f"未注册工具处理器: {name}")
            handler, access, safety_level, timeout_ms = handlers[name]
            self.tool_registry.register(
                definition,
                handler,
                access=access,
                safety_level=safety_level,
                timeout_ms=timeout_ms,
            )

    def tool_schemas(self) -> List[Dict]:
        """Return the live registry declarations for OpenAI tool calling."""
        return self.tool_registry.openai_tools()

    def execute_tool(self, name: str, arguments):
        """Execute a tool and preserve structured status for the orchestrator."""
        return self.tool_registry.execute(name, arguments)

    def _execute_tool(self, name: str, arguments) -> str:
        """Compatibility wrapper returning the JSON text sent back to the LLM."""
        result = self.execute_tool(name, arguments)
        logger.info(
            "Agent 工具调用: %s(%s) success=%s duration=%sms",
            name,
            str(arguments)[:80],
            result.success,
            result.duration_ms,
        )
        return result.content

    def _tool_current_detections(self, **_) -> Dict:
        frame_w, _frame_h = self._last_frame_size
        detections = self._current_detections()
        return {
            'detections': [
                self._det_item(item, frame_w)
                for item in detections
            ],
            'count': len(detections),
        }

    def _tool_scene_overview(self, **_) -> Dict:
        return self._scene_snapshot()

    def _tool_memory_summary(self, **_) -> Dict:
        # _memory_text/_event_text 已自守护；frames 计数在锁内读取
        with self._state_lock:
            frames = len(self.memory)
        return {
            'memory': self._memory_text(),
            'frames': frames,
            'events': self._event_text(limit=8),
        }

    def _tool_look_at_frame(self, question: str = '', **_) -> Dict:
        observation = self._call_vlm(
            str(question or '请描述画面中需要警惕的信息')
        )
        return {
            'observation': self._sanitize_observation(observation),
            'safety_note': '画面观察仅作辅助，不能作为通行授权',
        }

    def _tool_get_user_preferences(self, **_) -> Dict:
        return {'preferences': self._preferences()}

    def _tool_set_user_preference(self, key: str = '', value=None,
                                  **_) -> Dict:
        if self.context is not None and hasattr(
            self.context, 'set_preference'
        ):
            return self.context.set_preference(str(key or ''), value)
        return {'success': False, 'message': '偏好存储当前不可用'}

    def _tool_risk_assessment(self, **_) -> Dict:
        detections = self._current_detections()
        overall = self._overall_risk()
        urgency = assess_announcement(detections, overall)
        frame_w, _ = self._last_frame_size
        ranked = sorted(
            detections,
            key=lambda item: (
                RISK_ORDER.get(item.get('risk_level', 'safe'), 4),
                -float(item.get('confidence', 0) or 0),
                -float(item.get('area_ratio', 0) or 0),
            ),
        )
        counts = {level: 0 for level in (
            'critical', 'high', 'medium', 'low', 'safe'
        )}
        for item in detections:
            level = str(item.get('risk_level') or 'safe')
            counts[level] = counts.get(level, 0) + 1
        priority = []
        for item in ranked[:3]:
            entry = self._det_item(item, frame_w)
            reasons = []
            if item.get('trend') == 'approaching':
                reasons.append('正在接近')
            if float(item.get('area_ratio', 0) or 0) >= 0.05:
                reasons.append('距离较近')
            if item.get('light_state'):
                reasons.append(
                    f"灯色{STATE_CN.get(item['light_state'], item['light_state'])}"
                )
            entry['reasons'] = reasons or ['当前风险等级较高']
            priority.append(entry)
        return {
            'overall_risk': overall,
            'urgency_score': urgency.get('score', 0),
            'urgency': urgency.get('urgency', 'low'),
            'priority_reasons': urgency.get('reasons', []),
            'avoidance_priority': priority,
            'risk_counts': counts,
            'limitations': (
                '仅反映已检出目标；未检出、遮挡或模型漏检不等于安全，'
                '不能据此授权通行。'
            ),
        }

    def _tool_environment_trends(self, **_) -> Dict:
        detections = self._current_detections()
        approaching = [
            self._cn(item) for item in detections
            if item.get('trend') == 'approaching'
        ]
        receding = [
            self._cn(item) for item in detections
            if item.get('trend') == 'receding'
        ]
        # 锁内取事件与记忆快照、锁外统计，避免遍历中被管线线程追加
        with self._state_lock:
            events_snapshot = list(self.event_history)
            memory_frames = len(self.memory)
        event_counts: Dict[str, int] = {}
        for event in events_snapshot:
            key = str(event.get('type') or 'unknown')
            event_counts[key] = event_counts.get(key, 0) + 1
        return {
            'events': events_snapshot[-12:],
            'event_counts': event_counts,
            'approaching': approaching,
            'receding': receding,
            'memory_frames': memory_frames,
            'limitations': '事件由逐帧差异聚合，错检和漏检可能导致趋势偏差。',
        }

    def _tool_navigation_guidance(self, **_) -> Dict:
        detections = self._current_detections()
        overall = self._overall_risk()
        light = safety_policy.traffic_light_state(detections)
        frame_w, _ = self._last_frame_size
        ranked = sorted(
            detections,
            key=lambda item: (
                RISK_ORDER.get(item.get('risk_level', 'safe'), 4),
                -float(item.get('confidence', 0) or 0),
                -float(item.get('area_ratio', 0) or 0),
            ),
        )

        if light in ("red", "yellow", "unknown"):
            guidance_level = "stop"
            headline = {
                "red": "检测到红灯，请停步等待，不要通行",
                "yellow": "检测到黄灯，请停步等待，不要通行",
                "unknown": "信号灯状态无法确认，请停步等待",
            }[light]
        elif str(overall).lower() in ("critical", "high"):
            guidance_level = "stop"
            headline = "检测到高风险目标，请停步并用盲杖确认周围"
        elif ranked and ranked[0].get("risk_level") == "medium":
            guidance_level = "caution"
            headline = "前方存在障碍风险，请减速并用盲杖确认"
        elif ranked:
            guidance_level = "monitor"
            headline = "仅检测到低风险目标，请保持警惕"
        else:
            guidance_level = "uncertain"
            headline = "当前没有检测到目标，不能据此判断环境安全"

        zone_targets = {"左侧": [], "正前方": [], "右侧": []}
        for item in ranked[:6]:
            zone = self._zone(self._center_x(item), frame_w)
            zone_targets.setdefault(zone, []).append(self._cn(item))

        actions = ["保持盲杖触地探索，不因单次检测结果加快脚步"]
        if guidance_level == "stop":
            actions.insert(0, "停在当前位置，等待风险变化或人工协助")
        elif guidance_level == "caution":
            actions.insert(0, "降低速度，并优先让开正前方障碍")
        else:
            actions.insert(0, "继续观察，并关注左右两侧和目标运动趋势")
        if light == "green":
            actions.append("绿灯只表示检测到绿色区域，仍需人工确认路口")

        return {
            "guidance_level": guidance_level,
            "headline": headline,
            "actions": actions[:4],
            "zone_targets": {
                key: values for key, values in zone_targets.items() if values
            },
            "priority_targets": [
                self._det_item(item, frame_w) for item in ranked[:3]
            ],
            "safety_note": (
                "该建议只用于避障和风险提示，不包含过街或通行授权；"
                "漏检、遮挡和未检测区域不能视为安全。"
            ),
        }

    def _tool_system_health(self, **_) -> Dict:
        return self.status()

    def _tool_agent_capabilities(self, **_) -> Dict:
        return {
            'architecture': {
                'version': self.agent_version,
                'style': 'orchestrated-specialists-with-safety-guard',
                'max_tool_rounds': self.max_tool_rounds,
                'max_model_tool_calls': self.max_model_tool_calls,
                'max_specialist_roles': self.max_specialist_roles,
                'tool_prefetch_enabled': self.tool_prefetch_enabled,
                'specialist_execution_enabled': (
                    self.specialist_execution_enabled
                ),
            },
            'roles': [
                {
                    'key': role.key,
                    'name': role.name,
                    'description': role.description,
                    'authority': role.authority,
                    'preferred_tools': list(role.preferred_tools),
                }
                for role in SPECIALISTS
            ],
            'tools': self.tool_registry.capabilities(),
            'safety_rules': [
                '红灯、黄灯、未知灯色、高风险或无检测时，通行问题绕过 LLM',
                '绿灯不构成通行许可，只描述观测并要求人工确认',
                '主动播报和对话回复均经过确定性危险措辞拦截',
                '用户偏好只允许白名单键和范围校验后的写入',
            ],
        }

    def _tool_announcement_history(self, **_) -> Dict:
        with self._state_lock:
            announcements = list(self.announcement_history)
        return {
            'announcements': announcements[-8:],
            'count': len(announcements),
        }

    def _tool_alert_policy(self, **_) -> Dict:
        detections = self._current_detections()
        risk_level = self._overall_risk()
        candidates = self.should_announce(detections)
        urgency = assess_announcement(detections, risk_level)
        suppressed = len(detections) - len(candidates)

        if not detections:
            decision = 'silent'
            reasons = ['当前没有检测到目标']
        elif not candidates:
            decision = 'silent'
            reasons = [
                f"没有目标达到 min_level={self.min_level} 和 "
                f"min_confidence={self.min_confidence} 的播报门槛"
            ]
        else:
            decision = 'announce'
            reasons = list(urgency.get('reasons') or [])

        return {
            'decision': decision,
            'urgency': urgency.get('urgency', 'low'),
            'score': urgency.get('score', 0),
            'reasons': reasons,
            'announce_candidates': len(candidates),
            'suppressed_targets': max(0, suppressed),
            'min_level': self.min_level,
            'min_confidence': self.min_confidence,
            'cooldown_s': self.announce_cooldown,
            'last_announcement': self.last_announce_text,
            'safety_note': '静默只表示未达到主动播报门槛，不代表环境安全。',
        }

    def _det_item(self, d: Dict, frame_w: int) -> Dict:
        zone = self._zone(self._center_x(d), frame_w)
        item = {
            'name': self._cn(d),
            'risk': d.get('risk_level', 'low'),
            'zone': zone,
            'conf': round(float(d.get('confidence', 0)), 2),
        }
        if d.get('light_state'):
            item['light'] = STATE_CN.get(d['light_state'], d['light_state'])
        if d.get('trend') in TREND_CN and TREND_CN[d['trend']]:
            item['trend'] = TREND_CN[d['trend']]
        if d.get('track_age') is not None:
            item['duration_s'] = d['track_age']
        return item

    def _scene_snapshot(self) -> Dict:
        scene = None
        if self.context is not None and hasattr(self.context, 'get_scene'):
            try:
                scene = self.context.get_scene()
            except Exception:
                scene = None
        scene = scene or self._last_scene or {}
        keep = {}
        for k in ('scene_type', 'scene_type_cn', 'summary', 'object_count'):
            if k in scene:
                keep[k] = scene[k]
        if scene.get('movement_analysis'):
            keep['movement'] = scene['movement_analysis']
        by_class = (scene.get('object_distribution') or {}).get('by_class') or {}
        keep['object_counts'] = {k: v.get('count') for k, v in by_class.items()}
        keep['overall_risk'] = self._overall_risk()
        return keep

    # ==================== 多模态看图（VLM） ====================

    def _call_vlm(self, question: str) -> str:
        """调用多模态端点"看"当前画面，返回中文描述"""
        if not self.vlm_ok:
            return '（视觉语言模型未配置，无法直接查看画面）'
        jpeg = None
        if self.context is not None and hasattr(self.context, 'get_frame_jpeg'):
            try:
                jpeg = self.context.get_frame_jpeg()
            except Exception:
                jpeg = None
        if jpeg is None:
            return '（当前没有可用画面）'

        b64 = base64.b64encode(jpeg).decode('ascii')
        payload = {
            'model': self.vision_model,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': (
                    f"{question}\n"
                    "要求：用不超过 80 字的中文回答，只描述对视障人士出行重要的信息"
                    "（障碍物、文字标识、红绿灯状态、路况），不要描述无关细节。"
                )},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ]}],
            'temperature': 0.4,
            'max_tokens': 200,
            'stream': False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(
            f"{self.vision_base_url}/chat/completions", data=data, method='POST',
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Authorization': f'Bearer {self.vision_api_key}',
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.vlm_timeout) as r:
                body = json.loads(r.read().decode('utf-8'))
            content = body['choices'][0]['message']['content']
            return self._clean(content) or '（视觉模型未返回内容）'
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode('utf-8', errors='ignore')[:200]
            except Exception:
                pass
            logger.error(f"VLM HTTP 错误 {e.code}: {detail}")
        except Exception as e:
            logger.error(f"VLM 调用失败: {e}")
        return '（视觉模型调用失败，请稍后再试）'

    def _sanitize_observation(self, observation: str) -> str:
        """Prevent visual observations from becoming an implicit crossing grant."""
        sanitized = safety_policy.enforce_announcement_safety(
            self._clean(observation or ''),
            self._current_detections(),
        )
        if sanitized:
            return sanitized
        return '视觉模型给出了可能被理解为通行授权的表述，系统已拦截。'

    # ==================== LLM 调用（OpenAI 兼容） ====================

    def _call_llm(self, messages: List[Dict], max_tokens: int = 120,
                  tools: Optional[List[Dict]] = None, *,
                  temperature: Optional[float] = None,
                  timeout: Optional[float] = None,
                  record_failure: bool = True) -> Dict:
        """
        调用 OpenAI 兼容 /chat/completions。

        Args:
            temperature: 覆盖配置里的采样温度（规划器等要严格 JSON 时传 0）。
            timeout: 覆盖配置里的超时秒数（规划器用独立短超时，避免拖慢链路）。
            record_failure: 失败是否计入熔断。规划器传 False——
                它是可选的增强步骤，失败应当静默降级，不能把
                "在线" 徽标抖成 "本地回退"，也不该加速 30 秒熔断。

        Returns:
            assistant message 字典（含 content 与可选 tool_calls）；
            失败返回 {'content': ''}
        """
        if not self.llm_ok:
            return {'content': ''}
        # 熔断期内直接跳过网络调用，立即走回退路径
        if time.time() < self._fail_skip_until:
            return {'content': ''}

        request_timeout = self.timeout if timeout is None else float(timeout)
        request_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        use_tools = bool(tools) and self.tools_supported
        for _ in range(2):
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': request_temperature,
                'max_tokens': max_tokens,
                'stream': False,
            }
            if use_tools:
                payload['tools'] = tools
                payload['tool_choice'] = 'auto'
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, method='POST',
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Authorization': f'Bearer {self.api_key}',
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=request_timeout) as r:
                    body = json.loads(r.read().decode('utf-8'))
                msg = body['choices'][0]['message'] or {}
                self._fail_count = 0  # 调用成功，重置熔断计数
                self._llm_degraded = False
                self._last_error = ''
                return_msg = msg if isinstance(msg, dict) else {'content': str(msg)}
                return return_msg
            except urllib.error.HTTPError as e:
                detail = ''
                try:
                    detail = e.read().decode('utf-8', errors='ignore')[:300]
                except Exception:
                    pass
                logger.error(f"LLM HTTP 错误 {e.code}: {detail}")
                error = f"HTTP {e.code}: {detail[:200]}"
                # 端点不支持 tools 时自动降级为纯上下文模式，重试一次
                if use_tools and e.code in (400, 404, 422):
                    logger.warning("端点似乎不支持 tools，降级为纯上下文模式")
                    self.tools_supported = False
                    use_tools = False
                    continue
                break
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                error = f"{type(e).__name__}: {e}"
                break
        if not record_failure:
            logger.info(f"LLM 可选调用失败（不计入熔断）: {error}")
            return {'content': ''}
        self._register_failure(error)
        return {'content': ''}

    def _register_failure(self, error: str):
        """记录一次调用失败，连续失败达到阈值则触发熔断"""
        self._fail_count += 1
        self._llm_degraded = True
        self._last_error = error
        if self._fail_count >= self.FAIL_TRIP_THRESHOLD:
            self._fail_skip_until = time.time() + self.FAIL_SKIP_SECONDS
            logger.warning(
                f"LLM 连续失败 {self._fail_count} 次，"
                f"暂停调用 {self.FAIL_SKIP_SECONDS:.0f} 秒后自动恢复"
            )

    @staticmethod
    def _clean(text: str) -> str:
        """清理 LLM 输出的多余引号和空白（仅剥离首尾成对的引号）"""
        if not text:
            return ''
        text = text.strip()
        # 仅当同一引号在首尾“成对”出现时剥离，避免误删正文中合法的单侧
        # 引号（例如「警告」引用的开/闭引号只被剥掉一半）。
        # (开头, 结尾) 对：while 循环逐层剥除，支持三引号等多层包裹。
        wrappers = ('"', "'", '「」')
        changed = True
        while changed and len(text) >= 2:
            changed = False
            for pair in wrappers:
                open_q, close_q = pair[0], pair[-1]
                if text.startswith(open_q) and text.endswith(close_q):
                    text = text[1:-1].strip()
                    changed = True
                    break
        return text

    # ==================== 小工具 ====================

    def _cn(self, d: Dict) -> str:
        cn = d.get('class_name_cn')
        if cn:
            return cn
        return get_class_cn(d.get('class_name', ''), self._class_mapping)

    @staticmethod
    def _center_x(d: Dict) -> float:
        bbox = d.get('bbox') or [0, 0, 0, 0]
        if len(bbox) < 4:
            return 0.0
        return (bbox[0] + bbox[2]) / 2

    @staticmethod
    def _zone(center_x: float, frame_w: int) -> str:
        if not frame_w:
            return '正前方'
        return ZONE_CN[position_zone(center_x / frame_w)]

    def _current_detections(self) -> List[Dict]:
        if self.context is not None and hasattr(self.context, 'get_detections'):
            try:
                return self.context.get_detections() or []
            except Exception:
                pass
        return list(self._last_detections)

    def _overall_risk(self) -> str:
        if self.context is not None and hasattr(self.context, 'get_overall_risk'):
            try:
                return self.context.get_overall_risk() or 'safe'
            except Exception:
                pass
        return self._last_risk

    def _detection_health(self) -> Optional[Dict]:
        """检测推理健康；上下文未提供时返回 None（不编造健康状态）"""
        if self.context is not None and hasattr(
            self.context, 'get_detection_health'
        ):
            try:
                return self.context.get_detection_health() or None
            except Exception as e:
                logger.warning(f"读取检测健康状态失败: {e}")
        return None

    def _detection_health_warning(self) -> str:
        """推理异常时的提示前缀；健康时返回空串。

        「无目标」与「推理失败」必须区分：否则空场景会被说成
        “未检测到目标”，视障用户据此判断环境是危险的。
        """
        health = self._detection_health()
        if not health or not health.get('degraded'):
            return ''
        engines = health.get('engines') or []
        worst = max(
            (int(e.get('consecutive_errors', 0)) for e in engines), default=0
        )
        reasons = [e.get('last_error') for e in engines if e.get('last_error')]
        reason = reasons[-1] if reasons else '未知原因'
        return (
            f"【检测引擎异常】最近连续 {worst} 帧推理失败（{reason}）。"
            "以下检测结果可能不完整或为空，不能据此认定周围没有目标，"
            "也不能据此给出任何通行结论；请如实告知用户系统感知异常。\n"
        )

    def _preferences(self) -> Dict:
        if self.context is not None and hasattr(self.context, 'get_preferences'):
            try:
                return self.context.get_preferences() or {}
            except Exception:
                pass
        return {}

    # ==================== 状态 ====================

    def llm_available(self) -> bool:
        """当前是否健康可用；已配置但调用失败时返回 False。"""
        return bool(self.llm_ok and not self._llm_degraded)

    def last_orchestration(self) -> Dict:
        """返回最近一次对话的编排摘要，供前端展示真实执行证据。

        只暴露有界的展示字段：完整角色目录与调用审计仍走 /api/agent/status。
        这里回传的是已发生的执行记录（阶段、角色、工具、安全复核），
        不做任何重新推断或美化。
        """
        run = dict(self.orchestrator.last_run or {})
        if not run:
            return {}

        role_names = {role.key: role.name for role in SPECIALISTS}
        roles = []
        for item in run.get('role_contributions') or []:
            key = str(item.get('role') or '')
            roles.append({
                'role': key,
                'name': str(item.get('name') or role_names.get(key, key)),
                'status': str(item.get('status') or ''),
                'tool': str(item.get('tool') or ''),
                'summary': str(item.get('summary') or '')[:120],
            })

        tools = []
        for call in run.get('tool_calls') or []:
            tools.append({
                'phase': str(call.get('phase') or ''),
                'name': str(call.get('name') or ''),
                'success': bool(call.get('success')),
                'duration_ms': int(call.get('duration_ms') or 0),
                'cache_hit': bool(call.get('cache_hit')),
                'policy_denied': bool(call.get('policy_denied')),
            })

        lead_key = str(run.get('lead_role') or '')
        return {
            'intent': str(run.get('intent') or ''),
            'intent_confidence': float(run.get('intent_confidence') or 0),
            'intent_reason': str(run.get('intent_reason') or ''),
            'suggested_tools': [
                str(name) for name in (run.get('suggested_tools') or [])
            ],
            'lead_role': lead_key,
            'lead_role_name': role_names.get(lead_key, lead_key),
            'path': str(run.get('path') or ''),
            'path_reason': str(run.get('path_reason') or ''),
            'used_local_fallback': bool(run.get('used_local_fallback')),
            'plan_source': str(run.get('plan_source') or 'rules'),
            'planning': dict(run.get('planning') or {'source': 'rules'}),
            'duration_ms': int(run.get('duration_ms') or 0),
            'prefetch_tools': [
                str(name) for name in (run.get('prefetch_tools') or [])
            ],
            'stages': [dict(stage) for stage in (run.get('stages') or [])],
            'roles': roles,
            'tools': tools,
            'safety': dict(run.get('safety_review') or {}),
            'raw_reply': str(run.get('raw_reply') or '')[:300],
            'final_reply': str(run.get('sanitized_reply') or '')[:300],
        }

    def cockpit(self) -> Dict:
        """返回智能体驾驶舱所需的只读快照：角色、工具能力与调用审计。

        与 status() 分开，是因为驾驶舱按需拉取，不应进入前端 500ms 轮询。
        这里只做只读汇总，不执行任何工具。
        """
        orchestration = self.orchestrator.snapshot()
        last_run = dict(self.orchestrator.last_run or {})
        role_status = {
            str(item.get('role') or ''): str(item.get('status') or 'standby')
            for item in (last_run.get('role_contributions') or [])
        }
        roles = []
        for role in orchestration['roles']:
            roles.append({
                'key': role['key'],
                'name': role['name'],
                'description': role['description'],
                'authority': role['authority'],
                'preferred_tools': list(role['preferred_tools']),
                'last_status': role_status.get(role['key'], 'idle'),
            })
        return {
            'architecture': orchestration['architecture'],
            'orchestrator_version': orchestration['version'],
            'role_count': len(roles),
            'tool_count': len(self.tool_registry.names),
            'roles': roles,
            'tools': self.tool_registry.capabilities(),
            'recent_audit': self.tool_registry.recent_audit(10),
            'denied_recent': sum(
                1 for call in (last_run.get('tool_calls') or [])
                if call.get('policy_denied')
            ),
            'max_model_tool_calls': orchestration['max_model_tool_calls'],
            'max_specialist_roles': orchestration['max_specialist_roles'],
            'last_run': self.last_orchestration(),
        }

    def status(self) -> Dict:
        orchestration = self.orchestrator.snapshot()
        # 各共享结构计数在锁内一次性读取，避免与并发追加/清空交错
        with self._state_lock:
            memory_frames = len(self.memory)
            event_count = len(self.event_history)
            announcement_count = len(self.announcement_history)
            chat_turns = len(self.chat_history) // 2
        return {
            'agent_version': self.agent_version,
            'enabled': self.enabled,
            'llm_configured': self.llm_ok,
            'llm_active': self.llm_available(),
            'fallback_active': bool(self.enabled and not self.llm_available()),
            'last_error': self._last_error,
            'detection_health': self._detection_health(),
            'model': self.model if self.llm_available() else '本地回退',
            'tools_supported': self.tools_supported,
            'architecture': orchestration['architecture'],
            'orchestrator_version': orchestration['version'],
            'role_count': len(orchestration['roles']),
            'roles': [role['key'] for role in orchestration['roles']],
            'role_catalog': orchestration['roles'],
            'tool_count': len(self.tool_registry.names),
            'tool_names': self.tool_registry.names,
            'tool_prefetch_enabled': self.tool_prefetch_enabled,
            'specialist_execution_enabled': (
                self.specialist_execution_enabled
            ),
            'llm_planner_enabled': (
                self.llm_planner_enabled
                and self.orchestrator.llm_planner_enabled
            ),
            'llm_planner_version': self.orchestrator.planner.VERSION,
            'max_specialist_roles': self.max_specialist_roles,
            'max_model_tool_calls': self.max_model_tool_calls,
            'vision_active': self.vlm_ok,
            'vision_model': self.vision_model if self.vlm_ok else '',
            'memory_frames': memory_frames,
            'event_count': event_count,
            'announcement_count': announcement_count,
            'chat_turns': chat_turns,
            'last_plan': orchestration['last_plan'],
            'last_run': orchestration['last_run'],
            'recent_tool_calls': orchestration['recent_tool_calls'],
        }

    def reset(self):
        """Clear per-run state while preserving configuration and context."""
        # 四个共享结构的清空须与并发读/写互斥
        with self._state_lock:
            self.memory.clear()
            self.event_history.clear()
            self.announcement_history.clear()
            self.chat_history.clear()
        self._last_detections = []
        self._last_risk = 'safe'
        self._last_frame_size = (640, 480)
        self._last_scene = {}
        self.last_announce_time = 0.0
        self.last_announce_text = ''
        self._fail_count = 0
        self._fail_skip_until = 0.0
        self.tool_registry.clear_audit()
        self.orchestrator.reset()


def load_agent_config(config_path: str = 'config.yaml') -> Dict:
    """
    从 config.yaml（含 .env 环境变量覆盖）加载 agent 配置段。
    配置体系已统一到 ConfigManager，此处仅保留兼容入口。
    """
    return ConfigManager(config_path).get_section('agent')
