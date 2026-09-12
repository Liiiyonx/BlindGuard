# -*- coding: utf-8 -*-
"""
智能体核心模块

在传统 CV 管线（检测→风险→语音）之上叠加一层 LLM 推理，使系统具备：
1. 自然语言播报：用 LLM 生成简短口语化提示，替代固定模板
2. 多轮对话：用户可主动询问周围环境，Agent 调用检测结果回答
3. 时序记忆：滑动窗口保留最近若干帧的目标变化，喂给 LLM 做时序理解

LLM 走 OpenAI 兼容的 /chat/completions 接口（DeepSeek/通义千问/智谱GLM/OpenAI 均可），
仅用标准库 urllib 直连，不引入新依赖。未配置 api_key 或调用失败时自动回退到模板，
保证系统在离线/无密钥环境下仍可完整演示。
"""

import os
import json
import time
import logging
import threading
import urllib.request
import urllib.error
from collections import deque
from typing import List, Dict, Optional

logger = logging.getLogger('BlindGuard.Agent')

# 类别英文 -> 中文（导盲定制 6 类 + 常见类别兜底）
DEFAULT_CLASS_CN = {
    'car': '汽车', 'bicycle': '自行车', 'manhole_cover': '井盖',
    'public_facility': '公共设施', 'tactile_paving': '盲道',
    'traffic_light': '红绿灯', 'person': '行人', 'truck': '卡车',
    'bus': '公交车', 'motorcycle': '摩托车', 'dog': '狗', 'cat': '猫',
    'traffic light': '红绿灯', 'stop sign': '停车标志', 'fire hydrant': '消防栓',
}

# 风险等级排序，数值越小越紧急
RISK_ORDER = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'safe': 4}

SYSTEM_PROMPT = (
    "你是 BlindGuard 智能导盲系统的核心助手，服务对象是视障人士。"
    "你会收到摄像头每帧的目标检测结果（类别、风险等级、位置区域、面积占比）以及最近若干帧的记忆。"
    "任务一【播报】：生成一句简短、自然、口语化的中文语音提示，优先提醒高风险、近距离、正前方的目标；"
    "不超过 20 个字，不要堆砌细节，不要解释自己是 AI。"
    "任务二【对话】：回答用户关于周围环境的提问，回答不超过 80 字，基于检测结果如实描述，"
    "看不到的信息不要编造。始终使用中文。"
)


class BlindGuardAgent:
    """智能导盲 Agent：LLM 推理 + 时序记忆 + 对话"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.enabled = bool(config.get('enabled', True))

        llm = config.get('llm', {}) or {}
        self.base_url = (llm.get('base_url') or '').rstrip('/')
        self.api_key = llm.get('api_key') or ''
        self.model = llm.get('model') or 'deepseek-chat'
        self.temperature = float(llm.get('temperature', 0.6))
        self.timeout = float(llm.get('timeout', 8))

        ann = config.get('announce', {}) or {}
        self.announce_cooldown = float(ann.get('cooldown', 3.0))
        self.memory_size = int(ann.get('memory_size', 8))

        chat = config.get('chat', {}) or {}
        self.max_history = int(chat.get('max_history', 6))

        # 时序记忆：滑动窗口，每帧一个快照
        self.memory: deque = deque(maxlen=self.memory_size)
        # 对话历史 [(role, content), ...]
        self.chat_history: List[Dict] = []

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
            f"model={self.model} memory_size={self.memory_size}"
        )

    # ==================== 时序记忆 ====================

    def update_memory(self, detections: List[Dict],
                      risk_level: str = 'safe', frame_w: int = 640,
                      frame_h: int = 480):
        """每帧调用，更新滑动窗口记忆（不触发 LLM）"""
        snapshot = self._summarize_frame(detections, risk_level, frame_w, frame_h)
        self.memory.append(snapshot)

    def _summarize_frame(self, detections, risk_level, frame_w, frame_h):
        items = []
        denom = max(1, frame_w * frame_h)
        for d in detections:
            bbox = d.get('bbox', [0, 0, 0, 0])
            if len(bbox) < 4:
                bbox = [0, 0, 0, 0]
            cx = (bbox[0] + bbox[2]) / 2
            w = max(0, bbox[2] - bbox[0])
            h = max(0, bbox[3] - bbox[1])
            area_ratio = round((w * h) / denom, 4)
            if frame_w:
                if cx < frame_w * 0.25:
                    zone = '左'
                elif cx > frame_w * 0.75:
                    zone = '右'
                else:
                    zone = '正前'
            else:
                zone = '正前'
            cn = (d.get('class_name_cn')
                  or DEFAULT_CLASS_CN.get(d.get('class_name', ''), d.get('class_name', '目标')))
            items.append({
                'name': cn,
                'risk': d.get('risk_level', 'low'),
                'zone': zone,
                'area': area_ratio,
                'conf': round(float(d.get('confidence', 0)), 3),
            })
        return {'t': round(time.time(), 1), 'risk': risk_level, 'items': items}

    def _memory_text(self) -> str:
        if not self.memory:
            return '（暂无历史记忆）'
        lines = []
        n = len(self.memory)
        for i, snap in enumerate(self.memory):
            names = [f"{it['name']}/{it['risk']}/{it['zone']}"
                     for it in snap['items']]
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
        生成一句播报文本。使用已积累的记忆，不再次更新记忆
        （记忆由 update_memory 每帧维护，避免重复）。

        无目标时返回 None（由调用方决定是否播报"前方通畅"）。
        """
        if not detections:
            return None

        if not self.llm_ok:
            text = self._fallback_announce(detections)
        else:
            text = self._llm_announce(detections, risk_level)
            if not text:
                text = self._fallback_announce(detections)

        # 去抖：与上一次相同且未过冷却，则跳过
        now = time.time()
        if (text == self.last_announce_text
                and now - self.last_announce_time < self.announce_cooldown * 2):
            return None
        self.last_announce_time = now
        self.last_announce_text = text
        return text

    def _fallback_announce(self, detections) -> str:
        """无 LLM 时的模板播报（与原系统等价，但更自然一点）"""
        best = min(
            detections,
            key=lambda d: (RISK_ORDER.get(d.get('risk_level', 'low'), 3),
                           -float(d.get('confidence', 0)))
        )
        cn = (best.get('class_name_cn')
              or DEFAULT_CLASS_CN.get(best.get('class_name', ''), best.get('class_name', '目标')))
        level = best.get('risk_level', 'low')
        prefix = {'critical': '危险！', 'high': '危险！',
                  'medium': '注意，', 'low': '', 'safe': ''}.get(level, '')
        return f"{prefix}前方有{cn}"

    def _llm_announce(self, detections, risk_level) -> str:
        det_text = self._format_detections(detections)
        user = (
            f"当前帧检测结果：\n{det_text}\n"
            f"最近记忆：\n{self._memory_text()}\n"
            f"整体风险：{risk_level}\n"
            f"请生成一句不超过 20 字的中文语音播报。"
        )
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}]
        resp = self._call_llm(msgs, max_tokens=60)
        return self._clean(resp)

    def _format_detections(self, detections) -> str:
        if not detections:
            return '无目标'
        lines = []
        for d in detections:
            cn = (d.get('class_name_cn')
                  or DEFAULT_CLASS_CN.get(d.get('class_name', ''), d.get('class_name', '目标')))
            conf = float(d.get('confidence', 0))
            lines.append(
                f"- {cn}: 风险{d.get('risk_level', 'low')}, "
                f"置信度{conf:.0%}, 框{d.get('bbox', '')}"
            )
        return '\n'.join(lines)

    # ==================== 对话 ====================

    def chat(self, user_message: str, detections: Optional[List[Dict]] = None,
             risk_level: str = 'safe', frame_w: int = 640,
             frame_h: int = 480) -> str:
        """用户主动提问，返回 Agent 回复"""
        detections = detections or []

        if not self.llm_ok:
            return self._fallback_chat(user_message, detections)

        det_text = (self._format_detections(detections)
                    if detections else '（当前无活跃检测）')
        ctx = (
            f"用户提问：{user_message}\n"
            f"当前环境检测结果：\n{det_text}\n"
            f"最近记忆：\n{self._memory_text()}\n"
            f"整体风险：{risk_level}\n"
            f"请基于以上信息回答用户，不超过 80 字。"
        )
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        # 注入最近对话历史，保留多轮上下文
        for m in self.chat_history[-self.max_history * 2:]:
            msgs.append(m)
        msgs.append({'role': 'user', 'content': ctx})

        resp = self._call_llm(msgs, max_tokens=200)
        resp = self._clean(resp) or self._fallback_chat(user_message, detections)

        # 保存干净的问答对到历史
        self.chat_history.append({'role': 'user', 'content': user_message})
        self.chat_history.append({'role': 'assistant', 'content': resp})
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]
        return resp

    def _fallback_chat(self, user_message: str, detections) -> str:
        """无 LLM 时的本地对话：基于检测结果直接回答"""
        if not detections:
            return '当前没有看到周围目标，请把摄像头对准环境后再问。'
        names = []
        for d in detections:
            cn = (d.get('class_name_cn')
                  or DEFAULT_CLASS_CN.get(d.get('class_name', ''), d.get('class_name', '目标')))
            names.append(f"{cn}({d.get('risk_level', 'low')})")
        # 取最高风险目标做提醒
        worst = min(detections,
                    key=lambda d: RISK_ORDER.get(d.get('risk_level', 'low'), 3))
        wcn = (worst.get('class_name_cn')
               or DEFAULT_CLASS_CN.get(worst.get('class_name', ''), worst.get('class_name', '目标')))
        return f"我现在能看到：{'、'.join(names)}。其中 {wcn} 风险最高，请注意。"

    # ==================== LLM 调用（OpenAI 兼容） ====================

    def _call_llm(self, messages: List[Dict], max_tokens: int = 120) -> str:
        """调用 OpenAI 兼容的 /chat/completions 接口，失败返回空串"""
        if not self.llm_ok:
            return ''
        url = f"{self.base_url}/chat/completions"
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': max_tokens,
            'stream': False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(
            url, data=data, method='POST',
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Authorization': f'Bearer {self.api_key}',
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode('utf-8'))
            return body['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            logger.error(f"LLM HTTP 错误 {e.code}: {e.reason}")
            try:
                detail = e.read().decode('utf-8', errors='ignore')
                logger.error(f"LLM 错误详情: {detail[:300]}")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
        return ''

    @staticmethod
    def _clean(text: str) -> str:
        """清理 LLM 输出的多余引号和空白"""
        if not text:
            return ''
        text = text.strip()
        for q in ('"""', "'''", '"', "'", '「', '」'):
            text = text.strip().strip(q)
        return text.strip()

    # ==================== 状态 ====================

    def status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'llm_active': self.llm_ok,
            'model': self.model if self.llm_ok else '本地回退',
            'memory_frames': len(self.memory),
            'chat_turns': len(self.chat_history) // 2,
        }


def load_agent_config(config_path: str = 'config.yaml') -> Dict:
    """
    从 config.yaml（或环境变量）加载 agent 配置段。
    环境变量 BLINDGUARD_AGENT_LLM_API_KEY 等可覆盖文件配置。
    密钥等敏感配置请放在 .env（已 gitignore），此处自动加载。
    """
    # 先加载 .env 到环境变量（密钥不落配置文件）
    try:
        from dotenv import load_dotenv
        env_file = os.path.join(os.path.dirname(os.path.abspath(config_path)), '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            load_dotenv()  # 回退到默认查找
    except ImportError:
        pass

    cfg: Dict = {}
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full = yaml.safe_load(f) or {}
            cfg = full.get('agent', {}) or {}
        else:
            logger.warning(f"配置文件不存在: {config_path}，Agent 使用默认配置")
    except ImportError:
        logger.warning("PyYAML 未安装，Agent 使用默认配置")
    except Exception as e:
        logger.warning(f"读取 Agent 配置失败: {e}")

    # 环境变量覆盖
    env_map = {
        'BLINDGUARD_AGENT_ENABLED': ('enabled', _parse_bool),
        'BLINDGUARD_AGENT_LLM_BASE_URL': ('llm.base_url', str),
        'BLINDGUARD_AGENT_LLM_API_KEY': ('llm.api_key', str),
        'BLINDGUARD_AGENT_LLM_MODEL': ('llm.model', str),
        'BLINDGUARD_AGENT_ANNOUNCE_COOLDOWN': ('announce.cooldown', float),
        'BLINDGUARD_AGENT_ANNOUNCE_MEMORY_SIZE': ('announce.memory_size', int),
    }
    for env_key, (path, caster) in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            _set_nested(cfg, path, caster(val))
        except (ValueError, TypeError):
            logger.warning(f"环境变量 {env_key} 值非法: {val}")
    return cfg


def _parse_bool(v: str) -> bool:
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def _set_nested(d: Dict, path: str, value):
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
