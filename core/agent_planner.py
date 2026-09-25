# -*- coding: utf-8 -*-
"""LLM role planner for the BlindGuard agent.

规划器只在本地规则路由**没识别出意图**（``intent == "general"``）时启用，
且只决定"由哪些专家角色参与、顺序与理由"。

为什么不让 LLM 决定工具：``_run_specialist_team()`` 只执行
``role.preferred_tools[:1]``，``IntentPlan`` 里根本没有 tool 入口。
若让模型输出 ``{角色, 工具}``，计划证据会与实际 ``tool_calls`` 不一致——
**证据会撒谎**。只让模型选角色、工具由角色表推导，证据天然一致。

``intent`` / ``confidence`` / ``prefetch_tools`` / ``suggested_tools``
一律保持规则原值：一旦放权，模型就能把 intent 设成 ``preference``
（放行写工具）或 ``vision``（放行外呼视觉），整个权限模型会崩。

LLM 仍然没有通行授权权：通行问题根本不进入规划器（见编排层的硬门），
规划结果也必须通过本地白名单校验后才执行。
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("BlindGuard.Agent.Planner")

# 可被 LLM 指派执行的角色白名单。排除项及原因：
# - visual_observer：工具是 external，且 intent != "vision" 时
#   _run_specialist_team 会硬编码为 standby，指派它只会产生空转证据；
# - communicator：没有首选工具，由编排层在收尾阶段固定追加；
# - safety_guard：由确定性复核在收尾阶段追加，不能由模型决定何时把关。
PLANNER_ROLE_ALLOWLIST = (
    "perception",
    "risk_analyst",
    "navigation_advisor",
    "temporal_memory",
    "alert_manager",
    "self_monitor",
    "preference",
)

_ROLE_BRIEFS = {
    "perception": "读取当前目标、方位、置信度、跟踪趋势和红绿灯状态",
    "risk_analyst": "输出风险排序、风险原因和优先避让对象",
    "navigation_advisor": "给出停步、减速、盲杖确认等非通行避障建议",
    "temporal_memory": "聚合同一目标的出现、消失、接近、远离和灯色变化",
    "alert_manager": "解释播报门槛、紧急度、冷却与保守静默原因",
    "self_monitor": "报告模型、工具、视觉、降级与最近工具轨迹",
    "preference": "读取用户已保存的语速、播报详略偏好",
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
TRAFFIC_LIGHT_CLASSES = ("traffic_light", "traffic light")


@dataclass(frozen=True)
class RolePlan:
    """经本地校验后的角色计划。"""

    roles: Sequence[str]
    rationale: str
    dropped: Sequence[Sequence[str]] = ()

    @property
    def lead_role(self) -> str:
        return self.roles[0] if self.roles else ""

    @property
    def supporting_roles(self) -> Sequence[str]:
        return tuple(self.roles[1:])

    def as_dict(self) -> Dict:
        return {
            "roles": list(self.roles),
            "rationale": self.rationale,
            "dropped": [list(item) for item in self.dropped],
        }


def parse_plan_payload(content) -> Optional[Dict]:
    """从模型输出里提取计划 JSON。

    模型常包 ```json 围栏或前后加说明文字，因此只做
    "剥围栏 + 取首个 { 到末个 }" 的容错。解析失败一律返回 None，
    由调用方降级回规则计划——绝不猜测模型意图。
    """
    text = _CODE_FENCE.sub(" ", str(content or ""))
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_roles(raw_roles, *, max_roles: int = 3):
    """把模型给的角色序列收敛到可执行白名单。

    返回 ``(roles, dropped)``：
    - 白名单外的角色丢弃并逐条记入 ``dropped``（作为可核对证据）；
    - 去重、截断到 ``max_roles``，首位即主责角色；
    - 全部无效时返回空列表，由调用方降级到规则计划。
    """
    roles: List[str] = []
    dropped: List[List[str]] = []
    if not isinstance(raw_roles, (list, tuple)):
        return roles, dropped

    limit = max(1, int(max_roles))
    for item in raw_roles:
        key = str(item or "").strip().lower()
        if not key:
            continue
        if key not in PLANNER_ROLE_ALLOWLIST:
            dropped.append([key, "不在可指派角色白名单内"])
        elif key in roles:
            dropped.append([key, "重复指派"])
        elif len(roles) >= limit:
            dropped.append([key, "超出本轮角色数量上限"])
        else:
            roles.append(key)
    return roles, dropped


class LLMPlanner:
    """让 LLM 把"规则没认出来"的请求分解成专家角色。"""

    VERSION = "1.0"
    # 计划 JSON 很短（约 60 token），上限给 200 即可；
    # 给太大反而让本地小模型继续生成废话、更容易撞超时。
    MAX_TOKENS = 200

    def __init__(self, agent, *, max_roles: int = 3, timeout: float = 8.0):
        self.agent = agent
        self.max_roles = max(1, int(max_roles))
        self.timeout = max(0.5, float(timeout or 8.0))

    def build_prompt(self, user_message: str, detections: Sequence[Dict],
                     risk_level: str, rule_plan) -> List[Dict]:
        roster = "；".join(
            f"{key}（{_ROLE_BRIEFS[key]}）" for key in PLANNER_ROLE_ALLOWLIST
        )
        system = (
            "你是导盲系统的调度员。本地规则没有识别出用户意图，"
            "请你决定由哪些专家角色参与本轮处理，并给出顺序与理由。\n"
            f"可指派角色：{roster}。\n"
            f"最多指派 {self.max_roles} 个角色，列在第一位的是主责角色。\n"
            "只输出 JSON，不要输出任何其他内容，格式："
            '{"rationale": "一句话理由", "roles": ["perception", "risk_analyst"]}\n'
            "你不负责通行判断，不要输出任何通行许可或安全保证。"
        )
        # 现场概况只给计数与风险等级（枚举），不给模型生成的类别文本，
        # 避免把目标检测结果变成提示词注入面。
        detections = list(detections or [])
        lights = sum(
            1 for item in detections
            if item.get("class_name") in TRAFFIC_LIGHT_CLASSES
        )
        user = (
            f"用户提问：{user_message}\n"
            f"本地规则意图：{rule_plan.intent}\n"
            f"现场概况：检测到 {len(detections)} 个目标（其中红绿灯 {lights} 个），"
            f"整体风险等级 {risk_level}\n"
            "请选择最少的必要角色。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def propose(self, user_message: str, detections: Sequence[Dict],
                risk_level: str, rule_plan) -> Optional[RolePlan]:
        """返回经校验的角色计划；任何失败都返回 None 由调用方降级。"""
        message = self.agent._call_llm(
            self.build_prompt(user_message, detections, risk_level, rule_plan),
            max_tokens=self.MAX_TOKENS,
            temperature=0.0,
            timeout=self.timeout,
            record_failure=False,
        )
        payload = parse_plan_payload((message or {}).get("content") or "")
        if not payload:
            logger.info("规划器未返回可解析的计划，回落规则计划")
            return None

        roles, dropped = validate_roles(
            payload.get("roles"), max_roles=self.max_roles)
        if not roles:
            logger.info("规划器返回的角色全部无效，回落规则计划")
            return None
        return RolePlan(
            roles=tuple(roles),
            rationale=str(payload.get("rationale") or "")[:200],
            dropped=tuple(dropped),
        )
