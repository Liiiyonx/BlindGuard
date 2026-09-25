# -*- coding: utf-8 -*-
"""Architecture and capability tests for the orchestrated agent."""

import json
import os
import tempfile
import time
import unittest
import urllib.request

from core.agent import AGENT_TOOLS, BlindGuardAgent
from core.agent_orchestrator import IntentRouter, assess_announcement
from core.agent_tools import ToolRegistry
from core import safety_policy
from core.user_profile import UserProfile


class FakeContext:
    def __init__(self):
        self.detections = []
        self.risk = "safe"
        self.user_request_active = False
        self.preferences = {
            "voice_rate": 180,
            "announce_detail": "concise",
        }

    def is_user_request_active(self):
        return self.user_request_active

    def get_detections(self):
        return list(self.detections)

    def get_overall_risk(self):
        return self.risk

    def get_scene(self):
        return {
            "scene_type": "street",
            "scene_type_cn": "街道",
            "summary": "测试场景",
            "object_count": len(self.detections),
            "object_distribution": {"by_class": {}},
        }

    def get_frame_jpeg(self):
        return None

    def get_preferences(self):
        return dict(self.preferences)

    def set_preference(self, key, value):
        if key == "voice_rate":
            value = int(value)
        self.preferences[key] = value
        return {"success": True, "message": f"{key} 已设置为 {value}"}


def car(track_id=7, trend="approaching", risk="high"):
    return {
        "class_name": "car",
        "class_name_cn": "汽车",
        "confidence": 0.9,
        "risk_level": risk,
        "bbox": [280, 160, 500, 420],
        "area_ratio": 0.08,
        "track_id": track_id,
        "track_age": 1.5,
        "trend": trend,
    }


class AgentToolRegistryTests(unittest.TestCase):
    def test_llm_bypasses_system_proxy_by_default(self):
        # 本机系统代理会给每次 LLM 调用加 6-10 秒，默认必须直连
        agent = BlindGuardAgent({"enabled": True})

        self.assertFalse(agent.use_proxy)
        self.assertIsNot(agent._urlopen, urllib.request.urlopen)

    def test_llm_can_opt_into_system_proxy(self):
        agent = BlindGuardAgent({
            "enabled": True,
            "llm": {"use_proxy": True},
        })

        self.assertTrue(agent.use_proxy)
        self.assertIs(agent._urlopen, urllib.request.urlopen)

    def test_all_declarations_are_unique_and_registered(self):
        agent = BlindGuardAgent({"enabled": True})

        schema_names = [
            item["function"]["name"] for item in AGENT_TOOLS
        ]
        self.assertEqual(len(schema_names), 13)
        self.assertEqual(len(schema_names), len(set(schema_names)))
        self.assertEqual(schema_names, agent.tool_registry.names)
        self.assertIn("get_navigation_guidance", schema_names)
        self.assertIn("get_alert_policy", schema_names)

        capabilities = agent.tool_registry.capabilities()
        write_tools = [
            item["name"] for item in capabilities
            if item["access"] == "write"
        ]
        self.assertEqual(write_tools, ["set_user_preference"])

    def test_new_tools_return_structured_operational_data(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.update_memory(context.detections, "high", 640, 480)

        risk = json.loads(
            agent._execute_tool("get_risk_assessment", "{}")
        )
        trends = json.loads(
            agent._execute_tool("get_environment_trends", "{}")
        )
        health = json.loads(
            agent._execute_tool("get_system_health", "{}")
        )
        capabilities = json.loads(
            agent._execute_tool("get_agent_capabilities", "{}")
        )
        navigation = json.loads(
            agent._execute_tool("get_navigation_guidance", "{}")
        )
        alert_policy = json.loads(
            agent._execute_tool("get_alert_policy", "{}")
        )

        self.assertEqual(risk["overall_risk"], "high")
        self.assertTrue(risk["avoidance_priority"])
        self.assertIn("不能据此授权通行", risk["limitations"])
        self.assertIn("汽车", trends["approaching"])
        self.assertTrue(trends["events"])
        self.assertEqual(health["tool_count"], 13)
        self.assertEqual(
            health["architecture"],
            "orchestrated-specialists-with-safety-guard",
        )
        self.assertTrue(health["specialist_execution_enabled"])
        self.assertEqual(health["max_specialist_roles"], 3)
        self.assertEqual(health["max_model_tool_calls"], 6)
        self.assertEqual(len(capabilities["roles"]), 10)
        self.assertEqual(len(capabilities["tools"]), 13)
        roles = {
            role["key"]: role for role in capabilities["roles"]
        }
        self.assertIn("navigation_advisor", roles)
        self.assertIn("alert_manager", roles)
        self.assertIn(
            "get_navigation_guidance",
            roles["navigation_advisor"]["preferred_tools"],
        )
        self.assertIn(
            "get_alert_policy",
            roles["alert_manager"]["preferred_tools"],
        )
        self.assertEqual(navigation["guidance_level"], "stop")
        self.assertNotIn(
            "可以通行",
            json.dumps(navigation, ensure_ascii=False),
        )
        self.assertEqual(alert_policy["decision"], "announce")
        self.assertIn("不代表环境安全", alert_policy["safety_note"])

    def test_registry_rejects_missing_required_arguments(self):
        agent = BlindGuardAgent({"enabled": True})

        result = agent.execute_tool("set_user_preference", "{}")

        self.assertFalse(result.success)
        self.assertIn("缺少必填参数", result.error)

    def test_registry_enforces_tool_timeout(self):
        registry = ToolRegistry()
        registry.register(
            {
                "type": "function",
                "function": {
                    "name": "slow_tool",
                    "description": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            lambda **_: time.sleep(0.2) or {"ok": True},
            timeout_ms=20,
        )

        result = registry.execute("slow_tool", "{}")

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertIn("超时", result.error)

    def test_status_payload_is_flask_json_serializable(self):
        agent = BlindGuardAgent({"enabled": True})

        payload = json.dumps(agent.status(), ensure_ascii=False)

        self.assertIn('"tool_count": 13', payload)
        self.assertIn('"role_count": 10', payload)
        self.assertIn('"tool_prefetch_enabled": true', payload)
        self.assertIn('"specialist_execution_enabled": true', payload)
        self.assertIn('"max_model_tool_calls": 6', payload)


class AgentOrchestrationTests(unittest.TestCase):
    def test_local_preference_control_works_without_llm(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        agent.set_context(context)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("本地偏好控制不应调用 LLM")

        agent._call_llm = fail_if_called
        reply = agent.chat("请说慢一点", [], "safe")

        self.assertIn("设置", reply)
        self.assertEqual(context.preferences["voice_rate"], 160)
        self.assertEqual(agent.status()["last_run"]["path"], "local_control")

    def test_explicit_voice_rate_keeps_all_three_digits(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        agent.set_context(context)

        agent._call_llm = lambda *args, **kwargs: self.fail(
            "明确的语速设置不应调用 LLM"
        )
        reply = agent.chat("请把语速调到150", [], "safe")

        self.assertIn("150", reply)
        self.assertEqual(context.preferences["voice_rate"], 150)
        self.assertEqual(agent.status()["last_run"]["path"], "local_control")

    def test_capability_query_uses_registry_without_llm(self):
        agent = BlindGuardAgent({"enabled": True})
        agent._call_llm = lambda *args, **kwargs: self.fail(
            "能力查询不应调用 LLM"
        )

        reply = agent.chat("你会什么？", [], "safe")

        self.assertIn("13 个工具", reply)
        self.assertIn("确定性安全规则", reply)

    def test_intent_prefetch_is_bounded_traced_and_in_context(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.llm_ok = True
        captured = {}

        def fake_llm(messages, **kwargs):
            captured["messages"] = messages
            return {"content": "检测到汽车正在接近。"}

        agent._call_llm = fake_llm
        reply = agent.chat("前面有什么？", [], "safe")

        self.assertEqual(reply, "检测到汽车正在接近。")
        run = agent.status()["last_run"]
        self.assertEqual(run["path"], "llm_context")
        self.assertEqual(
            run["prefetch_tools"],
            ["get_risk_assessment", "get_current_detections"],
        )
        prefetch_calls = [
            call for call in run["tool_calls"]
            if call["phase"] == "prefetch"
        ]
        self.assertEqual(len(prefetch_calls), 2)
        self.assertEqual(
            len(run["role_contributions"]),
            5,
        )
        trace = list(agent.orchestrator.trace)
        self.assertEqual(
            len([
                entry for entry in trace
                if entry["phase"] == "prefetch"
            ]),
            2,
        )
        self.assertIn(
            "预取工具 get_current_detections",
            captured["messages"][-1]["content"],
        )
        self.assertIn(
            "汽车",
            captured["messages"][-1]["content"],
        )

    def test_model_tool_loop_is_traced_separately_from_prefetch(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.llm_ok = True
        agent.tools_supported = True
        responses = iter([
            {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_current_detections",
                        "arguments": "{}",
                    },
                }],
            },
            {"content": "汽车正在正前方接近，请停步观察。"},
        ])

        def fake_llm(messages, **kwargs):
            return next(responses)

        agent._call_llm = fake_llm
        reply = agent.chat("前面有什么？", [], "safe")

        self.assertEqual(reply, "汽车正在正前方接近，请停步观察。")
        run = agent.status()["last_run"]
        self.assertEqual(run["path"], "llm_tool_loop")
        self.assertEqual(
            [call["phase"] for call in run["tool_calls"]],
            ["prefetch", "prefetch", "specialist", "model"],
        )
        model_calls = [
            call for call in run["tool_calls"]
            if call["phase"] == "model"
        ]
        self.assertEqual(model_calls[0]["name"], "get_current_detections")
        self.assertTrue(model_calls[0]["cache_hit"])
        self.assertTrue(any(
            entry["phase"] == "model"
            and entry["name"] == "get_current_detections"
            for entry in agent.orchestrator.trace
        ))
        self.assertEqual(
            [stage["stage"] for stage in run["stages"]],
            [
                "routing",
                "prefetch",
                "specialists",
                "llm",
                "tool_loop",
                "safety_review",
            ],
        )
        self.assertEqual(run["stages"][3]["status"], "completed")
        self.assertEqual(run["stages"][4]["tool_count"], 1)

    def test_specialist_team_produces_real_role_evidence(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.llm_ok = True
        agent.tools_supported = True
        agent._call_llm = lambda *args, **kwargs: {
            "content": "汽车正在接近，请停步并用盲杖确认。"
        }

        reply = agent.chat("前面有什么风险？", [], "safe")
        run = agent.status()["last_run"]
        contributions = {
            item["role"]: item
            for item in run["role_contributions"]
        }

        self.assertIn("汽车", reply)
        self.assertEqual(
            contributions["perception"]["status"], "completed"
        )
        self.assertEqual(
            contributions["risk_analyst"]["status"], "completed"
        )
        self.assertEqual(
            contributions["navigation_advisor"]["status"], "completed"
        )
        self.assertEqual(
            contributions["navigation_advisor"]["tool"],
            "get_navigation_guidance",
        )
        self.assertEqual(
            contributions["safety_guard"]["source"],
            "deterministic_review",
        )
        specialist_stage = next(
            stage for stage in run["stages"]
            if stage["stage"] == "specialists"
        )
        self.assertGreaterEqual(specialist_stage["role_count"], 3)
        self.assertGreaterEqual(specialist_stage["tool_count"], 1)

    def test_model_cannot_write_preferences_outside_preference_intent(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.llm_ok = True
        agent.tools_supported = True
        responses = iter([
            {
                "content": "",
                "tool_calls": [{
                    "id": "policy-write",
                    "type": "function",
                    "function": {
                        "name": "set_user_preference",
                        "arguments": json.dumps({
                            "key": "voice_rate",
                            "value": "260",
                        }),
                    },
                }],
            },
            {"content": "当前不能修改偏好设置。"},
        ])
        agent._call_llm = lambda *args, **kwargs: next(responses)

        agent.chat("前面有什么风险？", [], "safe")
        run = agent.status()["last_run"]
        model_call = next(
            call for call in run["tool_calls"]
            if call["phase"] == "model"
        )

        self.assertTrue(model_call["policy_denied"])
        self.assertFalse(model_call["success"])
        self.assertIn("偏好", model_call["error"])
        self.assertEqual(context.preferences["voice_rate"], 180)

    def test_model_cannot_trigger_external_vision_without_vision_intent(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [car()]
        context.risk = "high"
        agent.set_context(context)
        agent.llm_ok = True
        agent.tools_supported = True
        responses = iter([
            {
                "content": "",
                "tool_calls": [{
                    "id": "policy-vision",
                    "type": "function",
                    "function": {
                        "name": "look_at_frame",
                        "arguments": json.dumps({
                            "question": "画面里有什么字？",
                        }, ensure_ascii=False),
                    },
                }],
            },
            {"content": "当前没有发起额外视觉调用。"},
        ])
        agent._call_llm = lambda *args, **kwargs: next(responses)
        agent._call_vlm = lambda *args, **kwargs: self.fail(
            "非视觉意图不应执行外部视觉工具"
        )

        agent.chat("前面有什么风险？", [], "safe")
        run = agent.status()["last_run"]
        model_call = next(
            call for call in run["tool_calls"]
            if call["phase"] == "model"
        )

        self.assertTrue(model_call["policy_denied"])
        self.assertFalse(model_call["success"])
        self.assertIn("视觉", model_call["error"])

    def test_safety_review_records_real_rewrite_evidence(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [{
            "class_name": "traffic_light",
            "class_name_cn": "红绿灯",
            "confidence": 0.95,
            "risk_level": "medium",
            "bbox": [250, 0, 390, 100],
            "light_state": "green",
        }]
        context.risk = "medium"
        agent.set_context(context)
        agent.orchestrator._finish_chat(
            user_message="现在能过马路吗？",
            reply="检测到绿灯，可以安全通行。",
            detections=context.detections,
            risk_level="medium",
            plan=agent.orchestrator.router.classify("现在能过马路吗？"),
            calls=[],
            path="test",
            started=time.perf_counter(),
        )

        run = agent.status()["last_run"]
        review = run["safety_review"]
        self.assertEqual(run["raw_reply"], "检测到绿灯，可以安全通行。")
        self.assertNotEqual(run["raw_reply"], run["sanitized_reply"])
        self.assertTrue(review["passage_intent"])
        self.assertTrue(review["raw_reply_dangerous"])
        self.assertTrue(review["guard_triggered"])
        # 整句替换必须被标记为 replaced，前端据此显示原始回复与保守答复的对照
        self.assertEqual(
            review["action"], safety_policy.REPLY_ACTION_REPLACED)
        self.assertEqual(review["appended"], "")
        self.assertTrue(review["safety_rewritten"])
        self.assertFalse(review["final_reply_dangerous"])
        self.assertTrue(review["idempotent"])
        self.assertTrue(review["guard_passed"])
        self.assertTrue(run["final_safe"])

    def test_safety_review_keeps_safe_reply_unchanged(self):
        agent = BlindGuardAgent({"enabled": True})
        agent.set_context(FakeContext())
        agent.orchestrator._finish_chat(
            user_message="前面有什么？",
            reply="检测到汽车正在接近。",
            detections=[car()],
            risk_level="high",
            plan=agent.orchestrator.router.classify("前面有什么？"),
            calls=[],
            path="test",
            started=time.perf_counter(),
        )

        run = agent.status()["last_run"]
        review = run["safety_review"]
        self.assertEqual(run["raw_reply"], run["sanitized_reply"])
        self.assertFalse(review["raw_reply_dangerous"])
        # 未改动时必须显式标记 unchanged，前端不展示任何对照块
        self.assertEqual(
            review["action"], safety_policy.REPLY_ACTION_UNCHANGED)
        self.assertEqual(review["appended"], "")
        self.assertFalse(review["safety_rewritten"])
        self.assertFalse(review["guard_triggered"])
        self.assertTrue(review["guard_passed"])
        self.assertTrue(run["final_safe"])

    def test_model_failure_is_recorded_as_local_fallback(self):
        # 模型中途不可用时链路必须如实标注本地兜底，
        # 不能一边显示"模型推理已完成"一边给的是离线模板回答。
        agent = BlindGuardAgent({
            "enabled": True,
            "llm": {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "test-key",
                "model": "unreachable",
            },
        })
        agent.set_context(FakeContext())
        reply = agent.chat("前面有什么？", [car()], "high")

        self.assertTrue(reply)
        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["path"], "llm_context")
        self.assertTrue(orchestration["used_local_fallback"])
        self.assertTrue(orchestration["path_reason"])

    def test_local_safety_path_is_not_marked_as_fallback(self):
        agent = BlindGuardAgent({"enabled": True})
        agent.chat("现在能过马路吗？", [car()], "high")

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration["path"], "deterministic_safety")
        self.assertFalse(orchestration["used_local_fallback"])

    def test_cockpit_exposes_roles_tools_and_audit(self):
        # 驾驶舱面板依赖这些字段，缺任何一个都会让台账显示不出来
        agent = BlindGuardAgent({"enabled": True})
        cockpit = agent.cockpit()

        self.assertEqual(cockpit["role_count"], 10)
        self.assertEqual(cockpit["tool_count"], 13)
        self.assertEqual(len(cockpit["roles"]), 10)
        self.assertEqual(len(cockpit["tools"]), 13)
        self.assertGreater(cockpit["max_model_tool_calls"], 0)
        for role in cockpit["roles"]:
            self.assertTrue(role["name"])
            self.assertTrue(role["authority"], "每个角色都必须声明权限边界")
            self.assertIn("last_status", role)
        for tool in cockpit["tools"]:
            self.assertIn(tool["access"], ("read", "write"))
            self.assertTrue(tool["safety_level"])
            self.assertGreater(tool["timeout_ms"], 0)

        access = {item["name"]: item["access"] for item in cockpit["tools"]}
        levels = {
            item["name"]: item["safety_level"] for item in cockpit["tools"]
        }
        self.assertEqual(access["set_user_preference"], "write")
        self.assertEqual(levels["look_at_frame"], "external")

    def test_navigation_advice_never_authorizes_crossing(self):
        agent = BlindGuardAgent({"enabled": True})
        context = FakeContext()
        context.detections = [{
            "class_name": "traffic_light",
            "class_name_cn": "红绿灯",
            "confidence": 0.95,
            "risk_level": "high",
            "bbox": [250, 0, 390, 100],
            "light_state": "red",
        }]
        context.risk = "high"
        agent.set_context(context)

        guidance = json.loads(
            agent._execute_tool("get_navigation_guidance", "{}")
        )
        text = json.dumps(guidance, ensure_ascii=False)

        self.assertEqual(guidance["guidance_level"], "stop")
        self.assertIn("不要通行", guidance["headline"])
        self.assertNotIn("可以通行", text)
        self.assertNotIn("放心走", text)
        self.assertIn("不包含过街或通行授权", text)

    def test_alert_policy_explains_conservative_silence(self):
        agent = BlindGuardAgent({"enabled": True})
        agent.set_context(FakeContext())

        policy = json.loads(
            agent._execute_tool("get_alert_policy", "{}")
        )

        self.assertEqual(policy["decision"], "silent")
        self.assertIn("未达到主动播报门槛", policy["safety_note"])
        self.assertIn("不代表环境安全", policy["safety_note"])

    def test_event_history_aggregates_trends_and_light_changes(self):
        agent = BlindGuardAgent({"enabled": True})
        first = [car(trend="stable")]
        red = [{
            "class_name": "traffic_light",
            "class_name_cn": "红绿灯",
            "confidence": 0.95,
            "risk_level": "high",
            "bbox": [250, 0, 390, 100],
            "light_state": "red",
            "track_id": 9,
        }]

        agent.update_memory(first, "high", 640, 480)
        agent.update_memory([car(trend="approaching")], "high", 640, 480)
        agent.update_memory(red, "high", 640, 480)

        event_types = [
            event["type"] for event in agent.event_history
        ]
        event_text = " ".join(
            event["text"] for event in agent.event_history
        )
        self.assertIn("approaching", event_types)
        self.assertIn("traffic_light", event_types)
        self.assertIn("接近", event_text)
        self.assertIn("红灯", event_text)

    def test_announcement_history_records_urgent_local_path(self):
        agent = BlindGuardAgent({"enabled": True})
        red = [{
            "class_name": "traffic_light",
            "class_name_cn": "红绿灯",
            "confidence": 0.95,
            "risk_level": "high",
            "bbox": [250, 0, 390, 100],
            "light_state": "red",
            "track_id": 9,
        }]

        announcement = agent.generate_announcement(red, "high")
        history = json.loads(
            agent._execute_tool("get_announcement_history", "{}")
        )

        self.assertIn("不要通行", announcement)
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["announcements"][0]["path"], "deterministic")
        self.assertGreaterEqual(
            history["announcements"][0]["score"], 75
        )
        self.assertTrue(any(
            "红灯" in reason
            for reason in history["announcements"][0]["reasons"]
        ))

    @staticmethod
    def _llm_ready_agent(context):
        """LLM 可用且拦截真实调用的 Agent，返回 (agent, 调用记录)"""
        agent = BlindGuardAgent({
            "enabled": True,
            "llm": {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "test-key",
                "model": "test-model",
            },
        })
        agent.set_context(context)
        calls = []

        def fake_call(messages, **kwargs):
            calls.append(kwargs)
            return {"content": "模型生成的播报"}

        agent._call_llm = fake_call
        return agent, calls

    def test_announcement_uses_llm_when_no_user_request(self):
        # 中风险、无红黄灯 → 走模型播报（高风险/红黄灯本就本地模板，不在此列）
        agent, calls = self._llm_ready_agent(FakeContext())

        announcement = agent.generate_announcement(
            [car(risk="medium")], "medium")
        history = json.loads(
            agent._execute_tool("get_announcement_history", "{}")
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("模型生成的播报", announcement)
        self.assertEqual(history["announcements"][0]["path"], "llm")

    def test_announcement_yields_llm_endpoint_while_user_asks(self):
        context = FakeContext()
        context.user_request_active = True
        agent, calls = self._llm_ready_agent(context)

        announcement = agent.generate_announcement(
            [car(risk="medium")], "medium")
        history = json.loads(
            agent._execute_tool("get_announcement_history", "{}")
        )

        # 提问期间不占用 LLM 端点，改用本地模板，并如实记录让出原因
        self.assertEqual(calls, [])
        self.assertTrue(announcement)
        self.assertEqual(
            history["announcements"][0]["path"], "yielded_to_user"
        )

    def test_intent_router_and_urgency_are_explainable(self):
        router = IntentRouter()
        self.assertEqual(
            router.classify("刚才发生了什么变化？").intent,
            "memory_trend",
        )
        urgency = assess_announcement([car()], "high")
        self.assertGreaterEqual(urgency["score"], 50)
        self.assertTrue(urgency["reasons"])

    def test_visual_observation_cannot_grant_crossing(self):
        agent = BlindGuardAgent({"enabled": True})

        safe_text = agent._sanitize_observation("可以安全通行，放心走")

        self.assertNotIn("放心走", safe_text)
        self.assertIn("拦截", safe_text)


class UserProfileValidationTests(unittest.TestCase):
    def test_unknown_preference_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = UserProfile(
                path=os.path.join(temp_dir, "user_profile.json")
            )

            ok, message = profile.set("arbitrary_key", "value")

            self.assertFalse(ok)
            self.assertIn("不支持的偏好项", message)
            self.assertIsNone(profile.get("arbitrary_key"))


if __name__ == "__main__":
    unittest.main()
