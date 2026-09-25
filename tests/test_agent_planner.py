# -*- coding: utf-8 -*-
"""LLM 角色规划器测试（不加载 Torch）。

规划器只在规则路由未命中（intent == "general"）时介入，且只决定角色。
这些用例同时守住"规划证据与实际执行一致"这条核心性质。
"""

import unittest

from core.agent import BlindGuardAgent
from core.agent_planner import (
    LLMPlanner,
    PLANNER_ROLE_ALLOWLIST,
    parse_plan_payload,
    validate_roles,
)

GENERAL_MESSAGE = "帮我看看现在是什么情况"
PLANNER_MARKER = "调度员"


def build_agent(**overrides):
    config = {
        "enabled": True,
        "llm": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "test-key",
            "model": "must-not-be-called",
        },
    }
    config.update(overrides)
    return BlindGuardAgent(config)


def green_light():
    return {
        "class_name": "traffic_light",
        "class_name_cn": "红绿灯",
        "confidence": 0.9,
        "risk_level": "medium",
        "bbox": [250, 0, 390, 100],
        "light_state": "green",
    }


def is_planner_call(messages) -> bool:
    return bool(messages) and PLANNER_MARKER in str(
        (messages[0] or {}).get("content") or "")


class ParsePlanPayloadTests(unittest.TestCase):
    def test_parses_plain_json(self):
        payload = parse_plan_payload('{"roles": ["perception"]}')
        self.assertEqual(payload, {"roles": ["perception"]})

    def test_strips_code_fence_and_surrounding_text(self):
        content = (
            "好的，我的安排如下：\n"
            "```json\n"
            '{"rationale": "需要风险排序", "roles": ["risk_analyst"]}\n'
            "```\n"
            "以上。"
        )
        payload = parse_plan_payload(content)
        self.assertEqual(payload["roles"], ["risk_analyst"])
        self.assertEqual(payload["rationale"], "需要风险排序")

    def test_returns_none_for_malformed_or_empty(self):
        for content in ("", None, "没有 JSON", "{不是合法 JSON}", "[1, 2]"):
            self.assertIsNone(parse_plan_payload(content))


class ValidateRolesTests(unittest.TestCase):
    def test_keeps_allowlisted_roles_in_order(self):
        roles, dropped = validate_roles(["perception", "risk_analyst"], max_roles=3)
        self.assertEqual(roles, ["perception", "risk_analyst"])
        self.assertEqual(dropped, [])

    def test_drops_roles_outside_allowlist(self):
        roles, dropped = validate_roles(
            ["perception", "car", "safety_guard"], max_roles=3)
        self.assertEqual(roles, ["perception"])
        self.assertEqual([item[0] for item in dropped], ["car", "safety_guard"])

    def test_deduplicates_and_truncates(self):
        roles, dropped = validate_roles(
            ["perception", "perception", "risk_analyst", "temporal_memory"],
            max_roles=2)
        self.assertEqual(roles, ["perception", "risk_analyst"])
        reasons = [item[1] for item in dropped]
        self.assertIn("重复指派", reasons)
        self.assertIn("超出本轮角色数量上限", reasons)

    def test_all_invalid_returns_empty(self):
        roles, _dropped = validate_roles(
            ["communicator", "visual_observer"], max_roles=3)
        self.assertEqual(roles, [])

    def test_non_sequence_is_rejected(self):
        self.assertEqual(validate_roles("perception", max_roles=3), ([], []))
        self.assertEqual(validate_roles(None, max_roles=3), ([], []))

    def test_allowlist_excludes_unexecutable_roles(self):
        # visual_observer(external/硬编码 standby)、communicator(无工具)、
        # safety_guard(收尾台账追加) 都不能被模型指派
        for role in ("visual_observer", "communicator", "safety_guard"):
            self.assertNotIn(role, PLANNER_ROLE_ALLOWLIST)


class PlannerUnitTests(unittest.TestCase):
    def test_propose_returns_none_when_no_valid_role(self):
        agent = build_agent()
        agent._call_llm = lambda *a, **kw: {
            "content": '{"rationale": "x", "roles": ["safety_guard"]}'
        }
        planner = LLMPlanner(agent)

        self.assertIsNone(planner.propose(GENERAL_MESSAGE, [], "safe",
                                          agent.orchestrator.router.classify(
                                              GENERAL_MESSAGE)))

    def test_prompt_uses_zero_temperature_short_timeout_and_no_breaker(self):
        agent = build_agent()
        seen = {}

        def fake_llm(messages, **kwargs):
            seen.update(kwargs)
            seen["planner_call"] = is_planner_call(messages)
            return {"content": '{"rationale": "x", "roles": ["perception"]}'}

        agent._call_llm = fake_llm
        planner = LLMPlanner(agent, timeout=2.5)
        plan = planner.propose(
            GENERAL_MESSAGE, [], "safe",
            agent.orchestrator.router.classify(GENERAL_MESSAGE))

        self.assertTrue(seen["planner_call"])
        self.assertEqual(seen["temperature"], 0.0)
        self.assertEqual(seen["timeout"], 2.5)
        self.assertFalse(seen["record_failure"])
        # 默认 max_tokens=120 会截断计划 JSON，必须显式放大
        self.assertGreater(seen["max_tokens"], 120)
        self.assertEqual(plan.roles, ("perception",))


class PlannerIntegrationTests(unittest.TestCase):
    def test_general_request_uses_llm_role_plan(self):
        agent = build_agent()

        def fake_llm(messages, **kwargs):
            if is_planner_call(messages):
                return {"content": (
                    '{"rationale": "描述模糊，先看风险再看变化",'
                    ' "roles": ["risk_analyst", "temporal_memory"]}')}
            return {"content": "我先说明一下当前情况。"}

        agent._call_llm = fake_llm
        agent.chat(GENERAL_MESSAGE, [], "safe")

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["planning"]["source"], "llm")
        self.assertEqual(orchestration["plan_source"], "llm")
        self.assertEqual(orchestration["lead_role"], "risk_analyst")
        self.assertIn("风险", orchestration["planning"]["rationale"])

        # 规划只改角色，工具/意图仍归规则所有
        self.assertEqual(orchestration["intent"], "general")
        self.assertEqual(orchestration["prefetch_tools"], ["get_scene_overview"])

        # 阶段数量与顺序不变（硬契约）
        self.assertEqual(
            [stage["stage"] for stage in orchestration["stages"]],
            ["routing", "prefetch", "specialists", "llm", "tool_loop",
             "safety_review"],
        )
        self.assertEqual(orchestration["stages"][0]["plan_source"], "llm")

        # 计划说的角色必须真的执行了，否则就是证据撒谎
        executed = [role["role"] for role in orchestration["roles"]]
        self.assertEqual(
            executed[:2], ["risk_analyst", "temporal_memory"])
        self.assertIn("communicator", executed)
        self.assertIn("safety_guard", executed)

    def test_malformed_plan_falls_back_to_rules(self):
        agent = build_agent()

        def fake_llm(messages, **kwargs):
            if is_planner_call(messages):
                return {"content": "我觉得应该先看风险"}
            return {"content": "我先说明一下当前情况。"}

        agent._call_llm = fake_llm
        agent.chat(GENERAL_MESSAGE, [], "safe")

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["planning"]["source"], "rules")
        self.assertEqual(orchestration["plan_source"], "rules")
        self.assertEqual(orchestration["lead_role"], "perception")

    def test_passage_question_never_plans(self):
        # 绿灯 + 中风险不会被确定性安全层旁路，会真实走到这段分支
        agent = build_agent()

        def fake_llm(messages, **kwargs):
            if is_planner_call(messages):
                raise AssertionError("通行问题不应进入 LLM 规划器")
            return {"content": "检测到绿灯，但不能确认路口安全。"}

        agent._call_llm = fake_llm
        reply = agent.chat("现在能过马路吗？", [green_light()], "medium")

        self.assertTrue(reply)
        self.assertEqual(agent.last_orchestration()["planning"]["source"],
                         "rules")

    def test_disabled_intent_router_never_plans(self):
        # 关闭规则路由时计划是内联的 general/0.5，不能被误判为"规则没认出"
        agent = build_agent()
        agent.orchestrator.intent_planner_enabled = False

        def fake_llm(messages, **kwargs):
            if is_planner_call(messages):
                raise AssertionError("规则路由关闭时不应启用规划器")
            return {"content": "我先说明一下当前情况。"}

        agent._call_llm = fake_llm
        agent.chat(GENERAL_MESSAGE, [], "safe")

        self.assertEqual(agent.last_orchestration()["planning"]["source"],
                         "rules")

    def test_environment_intent_never_plans(self):
        # 这条守住 stages 六项与 role_contributions==5 的既有断言
        agent = build_agent()

        def fake_llm(messages, **kwargs):
            if is_planner_call(messages):
                raise AssertionError("规则已识别意图时不应调用规划器")
            return {"content": "汽车正在正前方接近，请停步观察。"}

        agent._call_llm = fake_llm
        agent.chat("前面有什么？", [], "safe")

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["planning"]["source"], "rules")
        self.assertEqual(len(orchestration["roles"]), 5)

    def test_planner_failure_does_not_trip_breaker(self):
        # 端点不可达时，规划调用不得计入熔断：
        # 主循环本身会失败 2 次，若规划也被记录就会变成 3 次并触发 30 秒熔断。
        agent = build_agent()
        agent.chat(GENERAL_MESSAGE, [], "safe")

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["planning"]["source"], "rules")
        self.assertEqual(agent._fail_count, 2, "规划调用不应计入熔断计数")


if __name__ == "__main__":
    unittest.main()
