# -*- coding: utf-8 -*-
"""
智能体核心模块（V2.1）

在传统 CV 管线（检测→跟踪→风险→语音）之上叠加一层真正的 Agent：
1. 工具调用（Function Calling）：LLM 可自主调用 查检测/查场景/查记忆/看画面/
   读写用户偏好 等工具，能回答"现在能过马路吗"这类需要组合推理的问题，
   而不是只依赖塞进上下文的静态摘要
2. 自然语言播报：LLM 结合检测结果、红绿灯状态、跟踪趋势与时序记忆生成口语化提示
3. 多模态感知（可选）：接入 OpenAI 兼容的多模态端点（Qwen-VL/GLM-4V 等），
   Agent 需要时"亲眼看"画面，识别文字、标志与 YOLO 类别之外的异常
4. 用户偏好长期记忆：JSON 持久化（语速/播报详略），由应用层应用到设备
5. 离线兜底：未配置密钥或调用失败时回退模板播报与本地对话，系统完整可用

LLM 走 OpenAI 兼容 /chat/completions，仅用标准库 urllib，不引入新依赖。
"""

import os
import json
import time
import base64
import logging
import urllib.request
import urllib.error
from collections import deque
from typing import Dict, List, Optional

from core.config_manager import ConfigManager
from core.labels import get_class_cn
from core.traffic_light import STATE_CN

logger = logging.getLogger('BlindGuard.Agent')

# 风险等级排序，数值越小越紧急
RISK_ORDER = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'safe': 4}

TREND_CN = {'approaching': '正在接近', 'receding': '正在远离'}

SYSTEM_PROMPT = (
    "你是 BlindGuard 智能导盲系统的核心助手，服务对象是视障人士。"
    "你会收到摄像头检测出的目标（类别、风险等级、方位、红绿灯状态、运动趋势）以及最近若干帧的时序记忆。"
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
]

# 单轮对话最多允许的工具调用轮数
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

        chat = config.get('chat', {}) or {}
        self.max_history = int(chat.get('max_history', 6))

        # 端点是否支持 tools（首次报错自动降级为纯上下文模式）
        self.tools_supported = True

        # LLM 熔断：连续失败达到阈值后暂停调用一段时间，
        # 避免端点不可达时每次对话/播报都傻等超时
        self._fail_count = 0
        self._fail_skip_until = 0.0
        self.FAIL_TRIP_THRESHOLD = 3
        self.FAIL_SKIP_SECONDS = 30.0

        # 应用上下文（由 BlindGuardApp 注入，提供工具所需的数据源）
        self.context = None

        # 时序记忆：滑动窗口，每帧一个快照
        self.memory: deque = deque(maxlen=self.memory_size)
        # 对话历史 [(role, content), ...]
        self.chat_history: List[Dict] = []

        # 最后一帧的缓存（供无 context 时兜底，如冒烟测试直接实例化 Agent）
        self._last_detections: List[Dict] = []
        self._last_risk = 'safe'
        self._last_frame_size = (640, 480)
        self._last_scene: Dict = {}

        self.last_announce_time = 0.0
        self.last_announce_text = ''

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

    def _memory_text(self) -> str:
        if not self.memory:
            return '（暂无历史记忆）'
        lines = []
        n = len(self.memory)
        for i, snap in enumerate(self.memory):
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
        return '\n'.join(lines)

    # ==================== 播报生成 ====================

    def generate_announcement(self, detections: List[Dict],
                              risk_level: str = 'safe',
                              frame_w: int = 640,
                              frame_h: int = 480) -> Optional[str]:
        """
        生成一句播报文本。使用已积累的记忆，不再次更新记忆。

        无目标时返回 None（由调用方决定是否播报"前方通畅"）。
        """
        if not detections:
            return None

        detail = self._preferences().get('announce_detail', 'concise')
        if not self.llm_ok:
            text = self._fallback_announce(detections, frame_w)
        else:
            text = self._llm_announce(detections, risk_level, frame_w, detail)
            if not text:
                text = self._fallback_announce(detections, frame_w)

        # 去抖：与上一次相同且未过冷却，则跳过
        now = time.time()
        if (text == self.last_announce_text
                and now - self.last_announce_time < self.announce_cooldown * 2):
            return None
        self.last_announce_time = now
        self.last_announce_text = text
        return text

    def _fallback_announce(self, detections, frame_w: int = 640) -> str:
        """无 LLM 时的模板播报（含红绿灯与接近趋势）"""
        # 红灯优先：过马路场景的硬约束
        red = next((d for d in detections
                    if d.get('light_state') == 'red'
                    and d.get('risk_level') in ('critical', 'high', 'medium')), None)
        if red is not None:
            return "红灯，请等待绿灯再通行"

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
        if not self.llm_ok:
            return self._fallback_chat(user_message,
                                       detections if detections is not None
                                       else self._current_detections())

        if detections is None:
            detections = self._current_detections()

        det_text = (self._format_detections(detections, frame_w)
                    if detections else '（当前无活跃检测）')
        ctx = (
            f"用户提问：{user_message}\n"
            f"当前环境检测结果：\n{det_text}\n"
            f"最近记忆：\n{self._memory_text()}\n"
            f"整体风险：{risk_level}\n"
            "如以上信息不足以回答，请调用工具获取；请基于真实信息回答，不超过 80 字。"
        )
        msgs = [{'role': 'system', 'content': self._system_prompt()}]
        # 注入最近对话历史，保留多轮上下文
        msgs.extend(self.chat_history[-self.max_history * 2:])
        msgs.append({'role': 'user', 'content': ctx})

        reply = ''
        if self.tools_supported:
            for _ in range(MAX_TOOL_ROUNDS):
                msg = self._call_llm(msgs, max_tokens=300, tools=AGENT_TOOLS)
                calls = msg.get('tool_calls')
                if not calls:
                    reply = self._clean(msg.get('content') or '')
                    break
                # 原样回填 assistant 的工具调用消息，再逐个执行并回填结果
                msgs.append(msg)
                for tc in calls:
                    fn = tc.get('function') or {}
                    result = self._execute_tool(fn.get('name', ''),
                                                fn.get('arguments') or '{}')
                    logger.info(f"Agent 工具调用: {fn.get('name')}({str(fn.get('arguments'))[:80]})")
                    msgs.append({'role': 'tool',
                                 'tool_call_id': tc.get('id', ''),
                                 'content': result})

        if not reply:
            # 工具轮数用尽仍未产出文本，或端点不支持 tools：补一次无工具调用兜底
            final = self._call_llm(msgs, max_tokens=200)
            reply = self._clean(final.get('content') or '')
        if not reply:
            reply = self._fallback_chat(user_message, detections)

        # 保存干净的问答对到历史
        self.chat_history.append({'role': 'user', 'content': user_message})
        self.chat_history.append({'role': 'assistant', 'content': reply})
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]
        return reply

    def _fallback_chat(self, user_message: str, detections) -> str:
        """无 LLM 时的本地对话：基于检测结果直接回答"""
        if not detections:
            return '当前没有看到周围目标，请把摄像头对准环境后再问。'

        red = next((d for d in detections if d.get('light_state') == 'red'), None)
        green = next((d for d in detections if d.get('light_state') == 'green'), None)
        light_hint = ''
        if red is not None:
            light_hint = '红绿灯亮红灯，请等待绿灯再通行。'
        elif green is not None:
            light_hint = '红绿灯亮绿灯，注意观察后可通行。'

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

    def _execute_tool(self, name: str, arguments) -> str:
        """执行一次工具调用，返回给 LLM 的 JSON 字符串"""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        try:
            if name == 'get_current_detections':
                dets = self._current_detections()
                fw, _ = self._last_frame_size
                items = [self._det_item(d, fw) for d in dets]
                return json.dumps({'detections': items}, ensure_ascii=False)
            if name == 'get_scene_overview':
                return json.dumps(self._scene_snapshot(), ensure_ascii=False)
            if name == 'get_memory_summary':
                return json.dumps({'memory': self._memory_text()}, ensure_ascii=False)
            if name == 'look_at_frame':
                return self._call_vlm(str(args.get('question') or '请描述画面中需要警惕的信息'))
            if name == 'get_user_preferences':
                return json.dumps({'preferences': self._preferences()}, ensure_ascii=False)
            if name == 'set_user_preference':
                if self.context is not None and hasattr(self.context, 'set_preference'):
                    res = self.context.set_preference(str(args.get('key') or ''),
                                                      args.get('value'))
                else:
                    res = {'success': False, 'message': '偏好存储当前不可用'}
                return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {e}")
            return json.dumps({'error': str(e)}, ensure_ascii=False)
        return json.dumps({'error': f'未知工具: {name}'}, ensure_ascii=False)

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

    # ==================== LLM 调用（OpenAI 兼容） ====================

    def _call_llm(self, messages: List[Dict], max_tokens: int = 120,
                  tools: Optional[List[Dict]] = None) -> Dict:
        """
        调用 OpenAI 兼容 /chat/completions。

        Returns:
            assistant message 字典（含 content 与可选 tool_calls）；
            失败返回 {'content': ''}
        """
        if not self.llm_ok:
            return {'content': ''}
        # 熔断期内直接跳过网络调用，立即走回退路径
        if time.time() < self._fail_skip_until:
            return {'content': ''}

        use_tools = bool(tools) and self.tools_supported
        for _ in range(2):
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
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
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode('utf-8'))
                msg = body['choices'][0]['message'] or {}
                self._fail_count = 0  # 调用成功，重置熔断计数
                return msg if isinstance(msg, dict) else {'content': str(msg)}
            except urllib.error.HTTPError as e:
                detail = ''
                try:
                    detail = e.read().decode('utf-8', errors='ignore')[:300]
                except Exception:
                    pass
                logger.error(f"LLM HTTP 错误 {e.code}: {detail}")
                # 端点不支持 tools 时自动降级为纯上下文模式，重试一次
                if use_tools and e.code in (400, 404, 422):
                    logger.warning("端点似乎不支持 tools，降级为纯上下文模式")
                    self.tools_supported = False
                    use_tools = False
                    continue
                break
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                break
        self._register_failure()
        return {'content': ''}

    def _register_failure(self):
        """记录一次调用失败，连续失败达到阈值则触发熔断"""
        self._fail_count += 1
        if self._fail_count >= self.FAIL_TRIP_THRESHOLD:
            self._fail_skip_until = time.time() + self.FAIL_SKIP_SECONDS
            logger.warning(
                f"LLM 连续失败 {self._fail_count} 次，"
                f"暂停调用 {self.FAIL_SKIP_SECONDS:.0f} 秒后自动恢复"
            )

    @staticmethod
    def _clean(text: str) -> str:
        """清理 LLM 输出的多余引号和空白"""
        if not text:
            return ''
        text = text.strip()
        for q in ('"""', "'''", '"', "'", '「', '」'):
            text = text.strip().strip(q)
        return text.strip()

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
        if center_x < frame_w * 0.25:
            return '左侧'
        if center_x > frame_w * 0.75:
            return '右侧'
        return '正前方'

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

    def _preferences(self) -> Dict:
        if self.context is not None and hasattr(self.context, 'get_preferences'):
            try:
                return self.context.get_preferences() or {}
            except Exception:
                pass
        return {}

    # ==================== 状态 ====================

    def status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'llm_active': self.llm_ok,
            'model': self.model if self.llm_ok else '本地回退',
            'tools_supported': self.tools_supported,
            'tool_count': len(AGENT_TOOLS),
            'vision_active': self.vlm_ok,
            'vision_model': self.vision_model if self.vlm_ok else '',
            'memory_frames': len(self.memory),
            'chat_turns': len(self.chat_history) // 2,
        }


def load_agent_config(config_path: str = 'config.yaml') -> Dict:
    """
    从 config.yaml（含 .env 环境变量覆盖）加载 agent 配置段。
    配置体系已统一到 ConfigManager，此处仅保留兼容入口。
    """
    return ConfigManager(config_path).get_section('agent')
