# -*- coding: utf-8 -*-
"""Deterministic safety guardrails for crossing-related output.

The detector and LLM are probabilistic components. They may report what they
observe, but they must never authorize a visually impaired user to cross.
This module keeps that policy deterministic and independently testable.
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from core.labels import TRAFFIC_LIGHT_CLASSES

PASSAGE_INTENT_PATTERNS = (
    re.compile(r"(?:能不能|能否|可不可以|可以|能|是否).{0,4}(?:过马路|过街|通行|通过|过去|走)"),
    re.compile(r"(?:现在|当前|这里|路口).{0,5}(?:安全|能走|能过|可走|可过)"),
    re.compile(
        r"(?:前面|前方|这条|这条路|周边|周围|附近).{0,4}"
        r"(?:安全(?:吗|么)?|危险(?:吗|么))"
    ),
    re.compile(r"(?:过马路|过街|通行|过路口).{0,3}(?:吗|么|安全|可以|能)"),
)

# Positive authorization phrases. A match is ignored when its local context
# contains a refusal word such as "不能确认是否可通行".
DANGEROUS_APPROVAL_PATTERNS = (
    re.compile(
        r"(?:可以|可|能够|能|允许|建议|请|应当).{0,6}"
        r"(?:通行|通过|过马路|过去|走)"
    ),
    re.compile(r"(?:安全通行|放心走|放心通过|直接走|直接通行|走吧|可以通过)"),
    re.compile(r"(?:现在)\s*(?:安全)?\s*(?:通行|通过|过马路|过去|走)"),
    re.compile(
        r"(?:前方|环境|道路|路口|现在|周边|周围|附近)\s*(?:很|是)?\s*"
        r"(?:安全|无危险|没有危险|无风险|没有风险)"
    ),
    re.compile(r"(?:没有|无|未见|未发现)\s*(?:危险|车辆|障碍)"),
)

NEGATION_CONTEXT = re.compile(
    r"(?:不能|不得|不可|不要|无法|无|未|没|切勿|避免|是否|能否|能不能|不可以|难以)"
)
SAFETY_DISCLAIMER = re.compile(
    r"(?:不能确认|无法确认|不能保证|不能替代|请结合|请自行确认|人工判断|"
    r"以现场|仍需确认|辅助提示)"
)


@dataclass(frozen=True)
class PassageDecision:
    """Result of evaluating a user's crossing question."""

    is_passage_question: bool
    should_bypass_llm: bool
    light_state: Optional[str]
    reply: str


def has_passage_intent(text: str) -> bool:
    """Return whether the user is asking for a crossing/passage decision."""
    normalized = re.sub(r"\s+", "", text or "")
    return any(pattern.search(normalized) for pattern in PASSAGE_INTENT_PATTERNS)


def traffic_light_state(detections: Optional[Iterable[Dict]]) -> Optional[str]:
    """Return the most restrictive current light state.

    Red dominates yellow, yellow dominates green, and any detected traffic
    light with an unrecognized state is reported as ``unknown``.
    """
    states = []
    for detection in detections or []:
        if detection.get("class_name") not in TRAFFIC_LIGHT_CLASSES:
            continue
        state = str(detection.get("light_state") or "unknown").lower()
        states.append(state)

    for state in ("red", "yellow", "green"):
        if state in states:
            return state
    return "unknown" if states else None


# 授权类模式（DANGEROUS_APPROVAL_PATTERNS 下标 0~2）匹配“可以/建议/请……通行/走”。
# 真正的拒绝常把否定词夹在匹配文本内部（如“请等待，不要通行”“不能确认能否通行”），
# 因此这些模式的否定窗口必须包含匹配文本，否则会把确定性的红/黄灯“不要通行”误判为放行。
#
# 场景安全断言类模式（下标 3~4）匹配“周边没有危险”“前方无风险”“未发现车辆”等。
# 它们的匹配文本本身必然带有“没有/无/未”等词——那是“隐性安全保证”的组成部分，
# 而非对通行授权的拒绝。若把匹配文本计入否定窗口，这些词会永远命中 NEGATION_CONTEXT
# 而自我豁免，使这两条模式形同虚设（把“未检出即安全”的错觉放行给视障用户）。因此这
# 两条模式的否定窗口只取匹配【之前】的前缀；真正的拒绝（如“不能确认前方安全”）其否定
# 词仍位于前缀中，可正常豁免。
_SCENE_CLAIM_PATTERN_INDEXES = frozenset((3, 4))


def is_dangerous_approval(text: str) -> bool:
    """Detect language that authorizes crossing despite uncertain perception."""
    normalized = re.sub(r"\s+", "", text or "")
    for index, pattern in enumerate(DANGEROUS_APPROVAL_PATTERNS):
        for match in pattern.finditer(normalized):
            prefix = normalized[max(0, match.start() - 4):match.start()]
            if index in _SCENE_CLAIM_PATTERN_INDEXES:
                context = prefix
            else:
                context = prefix + match.group(0)
            if not NEGATION_CONTEXT.search(context):
                return True
    return False


def passage_reply(detections: Optional[Iterable[Dict]],
                  risk_level: str = "safe") -> str:
    """Build a conservative answer that never authorizes crossing."""
    detections = list(detections or [])
    state = traffic_light_state(detections)
    risk_level = str(risk_level or "safe").lower()

    if state == "red":
        return "检测到红灯，请等待，不要通行。系统不能替代盲杖和现场人工判断。"
    if state == "yellow":
        return "检测到黄灯，请等待，不要通行。系统不能替代盲杖和现场人工判断。"
    if risk_level in ("critical", "high"):
        return "当前检测到高风险目标，请停步等待。系统不能确认可以通行。"
    if not detections:
        return "当前没有可靠的路况和信号灯信息，不能判断能否通行。请先人工确认。"
    if state == "green":
        return "检测到绿灯，但仍不能确认路口是否安全，请结合盲杖、交通规则和人工判断。"
    return "无法可靠确认信号灯和路口状况，不能判断能否通行。请继续等待并人工确认。"


def evaluate_passage_question(user_message: str,
                              detections: Optional[Iterable[Dict]],
                              risk_level: str = "safe") -> PassageDecision:
    """Return a pre-LLM decision for crossing-related questions."""
    detections = list(detections or [])
    is_passage = has_passage_intent(user_message)
    state = traffic_light_state(detections)
    risk_level = str(risk_level or "safe").lower()
    # Green with a non-urgent scene may continue through the Agent loop so the
    # user still gets a contextual report. The final text is wrapped by
    # enforce_reply_safety and can never become an authorization.
    should_bypass = (
        is_passage
        and (
            not detections
            or state in ("red", "yellow", "unknown", None)
            or risk_level in ("critical", "high")
        )
    )
    return PassageDecision(
        is_passage_question=is_passage,
        should_bypass_llm=should_bypass,
        light_state=state,
        reply=passage_reply(detections, risk_level) if is_passage else "",
    )


REPLY_ACTION_UNCHANGED = "unchanged"
REPLY_ACTION_DISCLAIMER_APPENDED = "disclaimer_appended"
REPLY_ACTION_REPLACED = "replaced"

SAFETY_DISCLAIMER_SENTENCE = "。系统仅为辅助提示，不能确认可通行。"


@dataclass(frozen=True)
class ReplySafetyOutcome:
    """确定性安全后处理的结果：动作类型、最终文本、追加的说明。

    动作类型把“改了什么”变成可核对的证据：
    - unchanged：原样返回，未改动；
    - disclaimer_appended：保留原句，仅追加保守说明；
    - replaced：原句含授权措辞，整句替换为确定性保守答复。
    """

    action: str
    reply: str
    appended: str = ""


def review_reply_safety(user_message: str, reply: str,
                        detections: Optional[Iterable[Dict]],
                        risk_level: str = "safe") -> ReplySafetyOutcome:
    """Post-process an LLM reply and report how it was changed."""
    detections = list(detections or [])
    cleaned = (reply or "").strip()
    if is_dangerous_approval(cleaned):
        return ReplySafetyOutcome(
            REPLY_ACTION_REPLACED,
            passage_reply(detections, risk_level),
        )
    if not has_passage_intent(user_message):
        return ReplySafetyOutcome(REPLY_ACTION_UNCHANGED, cleaned)
    if not cleaned:
        return ReplySafetyOutcome(
            REPLY_ACTION_REPLACED,
            passage_reply(detections, risk_level),
        )
    if not SAFETY_DISCLAIMER.search(cleaned):
        return ReplySafetyOutcome(
            REPLY_ACTION_DISCLAIMER_APPENDED,
            cleaned.rstrip("。！! ") + SAFETY_DISCLAIMER_SENTENCE,
            SAFETY_DISCLAIMER_SENTENCE,
        )
    return ReplySafetyOutcome(REPLY_ACTION_UNCHANGED, cleaned)


def enforce_reply_safety(user_message: str, reply: str,
                         detections: Optional[Iterable[Dict]],
                         risk_level: str = "safe") -> str:
    """Post-process an LLM reply and remove any crossing authorization.

    只返回最终文本；需要知道“改了什么”的调用方使用 review_reply_safety。
    """
    return review_reply_safety(
        user_message, reply, detections, risk_level).reply


def enforce_announcement_safety(text: str,
                                detections: Optional[Iterable[Dict]]) -> str:
    """Prevent proactive announcements from granting permission to cross."""
    state = traffic_light_state(detections)
    if state == "red":
        return "红灯，请等待，不要通行"
    if state == "yellow":
        return "黄灯，请等待，不要通行"
    if is_dangerous_approval(text):
        return ""
    return (text or "").strip()


def should_use_local_announcement(detections: Optional[Iterable[Dict]],
                                  risk_level: str = "safe") -> bool:
    """Skip network-dependent LLM generation for urgent safety cues."""
    state = traffic_light_state(detections)
    return state in ("red", "yellow") or str(risk_level).lower() in (
        "critical", "high"
    )
