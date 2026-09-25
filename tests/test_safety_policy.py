# -*- coding: utf-8 -*-
"""Deterministic safety tests for crossing-related output."""

import unittest

from core import safety_policy
from core.agent import BlindGuardAgent


def traffic_light(state='red', risk_level='high'):
    return {
        'class_name': 'traffic_light',
        'class_name_cn': '红绿灯',
        'confidence': 0.9,
        'risk_level': risk_level,
        'bbox': [250, 0, 390, 100],
        'light_state': state,
    }


def high_risk_object():
    return {
        'class_name': 'car',
        'class_name_cn': '汽车',
        'confidence': 0.9,
        'risk_level': 'high',
        'bbox': [100, 100, 300, 300],
    }


class SafetyPolicyTests(unittest.TestCase):
    def test_red_question_bypasses_llm_and_denies_crossing(self):
        decision = safety_policy.evaluate_passage_question(
            '现在能过马路吗？', [traffic_light('red')], 'high')

        self.assertTrue(decision.is_passage_question)
        self.assertTrue(decision.should_bypass_llm)
        self.assertEqual(decision.light_state, 'red')
        self.assertIn('红灯', decision.reply)
        self.assertIn('不要通行', decision.reply)
        self.assertFalse(safety_policy.is_dangerous_approval(decision.reply))

    def test_yellow_question_bypasses_llm_and_denies_crossing(self):
        decision = safety_policy.evaluate_passage_question(
            '我能过去吗？', [traffic_light('yellow')], 'high')

        self.assertTrue(decision.should_bypass_llm)
        self.assertIn('黄灯', decision.reply)
        self.assertIn('不要通行', decision.reply)

    def test_green_is_descriptive_but_never_an_authorization(self):
        detections = [traffic_light('green', 'medium')]
        decision = safety_policy.evaluate_passage_question(
            '现在能过马路吗？', detections, 'medium')
        reply = safety_policy.passage_reply(detections, 'medium')

        self.assertTrue(decision.is_passage_question)
        self.assertFalse(decision.should_bypass_llm)
        self.assertIn('绿灯', reply)
        self.assertIn('不能确认', reply)
        self.assertNotIn('可以通行', reply)
        self.assertFalse(safety_policy.is_dangerous_approval(reply))

    def test_unknown_light_or_no_detection_cannot_judge(self):
        unknown = safety_policy.evaluate_passage_question(
            '这里安全吗？', [traffic_light('unknown', 'medium')], 'medium')
        empty = safety_policy.evaluate_passage_question(
            '这里安全吗？', [], 'safe')

        for decision in (unknown, empty):
            self.assertTrue(decision.should_bypass_llm)
            self.assertIn('不能判断', decision.reply)
            self.assertFalse(
                safety_policy.is_dangerous_approval(decision.reply))

    def test_high_risk_scene_forces_stop_even_without_light(self):
        decision = safety_policy.evaluate_passage_question(
            '前面安全吗？', [high_risk_object()], 'high')

        self.assertTrue(decision.should_bypass_llm)
        self.assertIn('高风险', decision.reply)
        self.assertIn('停步等待', decision.reply)

    def test_dangerous_llm_reply_is_replaced(self):
        detections = [traffic_light('green', 'medium')]
        for original in (
            '检测到绿灯，可以安全通行。',
            '看到绿灯即可通行。',
            '建议现在直接走到对面。',
        ):
            self.assertTrue(safety_policy.is_dangerous_approval(original))
            reply = safety_policy.enforce_reply_safety(
                '现在能过马路吗？', original, detections, 'medium')
            self.assertFalse(safety_policy.is_dangerous_approval(reply))
            self.assertIn('不能确认', reply)

    def test_safe_refusal_keeps_its_wording(self):
        detections = [traffic_light('green', 'medium')]
        original = '不能确认能否通行，请人工判断。'

        reply = safety_policy.enforce_reply_safety(
            '现在能过马路吗？', original, detections, 'medium')

        self.assertEqual(reply, original)

    def test_non_passage_question_is_not_rewritten(self):
        original = '检测到行人，位于正前方。'

        reply = safety_policy.enforce_reply_safety(
            '前面有什么？', original, [high_risk_object()], 'high')

        self.assertEqual(reply, original)

    def test_authorization_is_blocked_even_without_passage_question(self):
        original = '前方没有危险，您可以放心走。'

        reply = safety_policy.enforce_reply_safety(
            '前面有什么？', original, [high_risk_object()], 'high')

        self.assertNotEqual(reply, original)
        self.assertFalse(safety_policy.is_dangerous_approval(reply))
        self.assertIn('不能确认', reply)

    def test_hidden_safety_claim_is_dangerous(self):
        # 严重1回归：隐性安全保证（“未检出即安全”的错觉）必须判为危险措辞。
        # 这些短语的匹配文本自带“没有/无/未”，修复前会因此命中否定词而自我豁免放行。
        for text in (
            '周边没有危险',
            '附近未发现车辆',
            '路口未见车辆',
            '没有障碍物',
            '前面很安全可以放心通行',
        ):
            self.assertTrue(
                safety_policy.is_dangerous_approval(text),
                f'应判为危险措辞: {text}')

    def test_negation_exemption_is_not_overridden(self):
        # 场景断言仍判危险：匹配文本内的“无”不得借否定窗口自我豁免。
        self.assertTrue(safety_policy.is_dangerous_approval('前方无危险'))
        # 真实拒绝/存疑仍豁免（保留原措辞）：否定词可在匹配文本内部或前缀中；
        # “未能确认……”靠前缀的“未”识别为存疑，不得被过度收紧。
        for text in (
            '不能确认能否通行，请人工判断。',
            '红灯，请等待，不要通行',
            '请不要通行',
            '无法判断是否可以过马路',
            '未能确认可以通行',
        ):
            self.assertFalse(
                safety_policy.is_dangerous_approval(text),
                f'应豁免(非授权): {text}')

    def test_general_danger_query_is_not_treated_as_crossing_request(self):
        for question in ('前面有什么危险？', '周围有什么危险？'):
            self.assertFalse(safety_policy.has_passage_intent(question))
            self.assertEqual(
                safety_policy.enforce_reply_safety(
                    question, '检测到汽车正在接近。', [], 'high'),
                '检测到汽车正在接近。',
            )

    def test_urgent_announcements_are_local_and_cannot_authorize(self):
        for state in ('red', 'yellow'):
            detections = [traffic_light(state)]
            self.assertTrue(
                safety_policy.should_use_local_announcement(
                    detections, 'high'))
            text = safety_policy.enforce_announcement_safety(
                '可以安全通行', detections)
            self.assertIn('不要通行', text)
            self.assertFalse(safety_policy.is_dangerous_approval(text))

        self.assertTrue(
            safety_policy.should_use_local_announcement(
                [high_risk_object()], 'critical'))
        self.assertEqual(
            safety_policy.enforce_announcement_safety(
                '放心走，路上没有车辆', [high_risk_object()]),
            '',
        )

    def test_reply_safety_outcome_reports_what_changed(self):
        # 前端要把“改了什么”显示出来，动作类型即该契约。
        detections = [traffic_light('green', 'medium')]

        replaced = safety_policy.review_reply_safety(
            '现在能过马路吗？', '检测到绿灯，可以安全通行。', detections, 'medium')
        self.assertEqual(
            replaced.action, safety_policy.REPLY_ACTION_REPLACED)
        self.assertIn('不能确认', replaced.reply)
        self.assertEqual(replaced.appended, '')
        self.assertFalse(safety_policy.is_dangerous_approval(replaced.reply))

        appended = safety_policy.review_reply_safety(
            '现在能过马路吗？', '检测到绿灯。', detections, 'medium')
        self.assertEqual(
            appended.action, safety_policy.REPLY_ACTION_DISCLAIMER_APPENDED)
        self.assertEqual(
            appended.appended, safety_policy.SAFETY_DISCLAIMER_SENTENCE)
        # 追加分支必须保留原句，只在其后补保守说明
        self.assertTrue(appended.reply.startswith('检测到绿灯。'))
        self.assertTrue(appended.reply.endswith(appended.appended))

        untouched = safety_policy.review_reply_safety(
            '前面有什么？', '检测到汽车正在接近。', [high_risk_object()], 'high')
        self.assertEqual(
            untouched.action, safety_policy.REPLY_ACTION_UNCHANGED)
        self.assertEqual(untouched.reply, '检测到汽车正在接近。')
        self.assertEqual(untouched.appended, '')

    def test_review_outcome_matches_reply_wrapper(self):
        # 旧接口必须与新接口给出同一最终文本，避免两条路径漂移。
        detections = [traffic_light('green', 'medium')]
        for question, original in (
            ('现在能过马路吗？', '检测到绿灯。'),
            ('现在能过马路吗？', '检测到绿灯，可以安全通行。'),
            ('前面有什么？', '检测到汽车正在接近。'),
            ('现在能过马路吗？', ''),
        ):
            outcome = safety_policy.review_reply_safety(
                question, original, detections, 'medium')
            self.assertEqual(
                outcome.reply,
                safety_policy.enforce_reply_safety(
                    question, original, detections, 'medium'),
                f'两条路径文本不一致: {question} / {original}')

    def test_passage_refusal_still_gives_non_passage_guidance(self):
        # 复合问题（能否过街 + 该往哪站）必须先确定性拒绝通行，
        # 再用导航工具补一段非通行建议，而不是只回一句“不能通行”。
        agent = BlindGuardAgent({
            'enabled': True,
            'llm': {
                'base_url': 'http://127.0.0.1:9/v1',
                'api_key': 'test-key',
                'model': 'must-not-be-called',
            },
        })

        def fail_if_called(*args, **kwargs):
            raise AssertionError('确定性安全路径不应调用 LLM')

        agent._call_llm = fail_if_called
        detections = [traffic_light('red')]
        # 导航工具读取的是 Agent 自己的逐帧状态，而不是 chat() 的入参
        agent.update_memory(detections, 'high', 640, 480)
        reply = agent.chat('现在能过马路吗？不能的话我该往哪边站', detections, 'high')

        self.assertIn('不要通行', reply)
        self.assertIn('当前方位：正前方红绿灯', reply)
        self.assertIn('停在当前位置', reply)
        self.assertIn('不包含过街或通行授权', reply)
        self.assertFalse(safety_policy.is_dangerous_approval(reply))

        orchestration = agent.last_orchestration()
        self.assertEqual(orchestration['path'], 'deterministic_safety')
        # 决策依据必须解释真实走的路径，不能与路径自相矛盾
        self.assertIn('确定性安全规则', orchestration['path_reason'])
        self.assertIn(
            'get_navigation_guidance',
            [tool['name'] for tool in orchestration['tools']],
            '复合问题应留下导航工具的真实调用记录')

    def test_plain_passage_question_skips_navigation_guidance(self):
        # 没有夹带非通行求助时不应多调导航工具，保持原有确定性路径
        agent = BlindGuardAgent({
            'enabled': True,
            'llm': {
                'base_url': 'http://127.0.0.1:9/v1',
                'api_key': 'test-key',
                'model': 'must-not-be-called',
            },
        })
        reply = agent.chat('现在能过马路吗？', [traffic_light('red')], 'high')

        self.assertIn('不要通行', reply)
        self.assertNotIn('停在当前位置', reply)
        self.assertEqual(agent.last_orchestration()['tools'], [])

    def test_agent_chat_bypasses_llm_for_red_light(self):
        agent = BlindGuardAgent({
            'enabled': True,
            'llm': {
                'base_url': 'http://127.0.0.1:9/v1',
                'api_key': 'test-key',
                'model': 'must-not-be-called',
            },
        })

        def fail_if_called(*args, **kwargs):
            raise AssertionError('红灯问题不应调用 LLM')

        agent._call_llm = fail_if_called
        reply = agent.chat(
            '现在能过马路吗？', [traffic_light('red')], 'high')

        self.assertIn('不要通行', reply)
        self.assertEqual(agent.chat_history[-1]['content'], reply)

    def test_agent_red_announcement_uses_local_path(self):
        agent = BlindGuardAgent({
            'enabled': True,
            'llm': {
                'base_url': 'http://127.0.0.1:9/v1',
                'api_key': 'test-key',
                'model': 'must-not-be-called',
            },
        })

        def fail_if_called(*args, **kwargs):
            raise AssertionError('红灯主动播报不应调用 LLM')

        agent._call_llm = fail_if_called
        announcement = agent.generate_announcement(
            [traffic_light('red')], 'high')

        self.assertIn('红灯', announcement)
        self.assertIn('不要通行', announcement)


if __name__ == '__main__':
    unittest.main()
