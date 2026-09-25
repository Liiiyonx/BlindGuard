# -*- coding: utf-8 -*-
"""
工具调用循环端到端测试：
用本地模拟的 OpenAI 兼容端点验证 Agent 的"感知-推理-行动"闭环，
无需真实 API Key 即可确定性测试 function calling 链路与偏好生效。

模拟脚本：
  第 1 轮: LLM 请求调用 get_current_detections 工具
  第 2 轮: LLM 基于工具结果给出最终回答

偏好调整通过独立的本地意图路由验证，确保写工具只在明确偏好意图下执行。
"""
import os

os.environ['BLINDGUARD_AGENT_LLM_BASE_URL'] = 'http://127.0.0.1:8765/v1'
os.environ['BLINDGUARD_AGENT_LLM_API_KEY'] = 'test-key'
os.environ['BLINDGUARD_AGENT_LLM_MODEL'] = 'mock-chat'

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

REQUESTS = []


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        REQUESTS.append(body)
        n = len(REQUESTS)
        if n == 1:
            msg = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_1', 'type': 'function',
                 'function': {'name': 'get_current_detections', 'arguments': '{}'}}]}
        else:
            msg = {'role': 'assistant',
                   'content': '前方检测到车辆和红灯，请保持等待。'}
        resp = {'choices': [{'message': msg}]}
        data = json.dumps(resp).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    server = HTTPServer(('127.0.0.1', 8765), MockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    from app import BlindGuardApp
    app_obj = BlindGuardApp()
    assert app_obj.agent.llm_ok, 'mock LLM 应可用'
    assert app_obj.agent.base_url.endswith(':8765/v1'), \
        f"环境变量覆盖未生效: {app_obj.agent.base_url}"

    # 注入假的当前检测结果（模拟红绿灯红灯帧）
    fake = [{'class_name': 'traffic_light', 'class_name_cn': '红绿灯',
             'confidence': 0.9, 'risk_level': 'high',
             'bbox': [250, 0, 390, 100], 'area_ratio': 0.03,
             'light_state': 'red', 'track_id': 7, 'track_age': 1.2,
             'trend': 'stable'}]
    with app_obj.detections_lock:
        app_obj.detections = fake
    app_obj.overall_risk = 'high'

    reply = app_obj.agent.chat('前面有什么？')

    # 环境事件闭环只应发生“只读工具调用 -> 最终回答”两轮请求。
    assert len(REQUESTS) == 2, f"期望 2 轮请求，实际 {len(REQUESTS)}"
    tools = REQUESTS[0].get('tools') or []
    assert len(tools) == 13, f"第 1 轮应声明 13 个工具，实际 {len(tools)}"
    names = [t['function']['name'] for t in tools]
    assert 'look_at_frame' in names, '应包含看图工具'
    assert 'get_risk_assessment' in names, '应包含结构化风险工具'
    assert 'get_navigation_guidance' in names, '应包含非通行导航建议工具'
    assert 'get_environment_trends' in names, '应包含时序事件工具'
    assert 'get_alert_policy' in names, '应包含播报策略工具'
    assert 'get_agent_capabilities' in names, '应包含能力查询工具'

    # 环境问题在首次 LLM 请求前已预取有界本地只读数据。
    prefetch_context = REQUESTS[0]['messages'][-1]['content']
    assert '预取工具 get_current_detections' in prefetch_context, \
        f"首轮上下文应包含预取检测数据: {prefetch_context[:300]}"
    assert '红灯' in prefetch_context, '预取检测数据应包含红灯状态'

    # 第 2 轮消息中应包含第 1 轮的工具结果（检测列表，含红灯）
    tool_msgs_r2 = [m for m in REQUESTS[1]['messages'] if m.get('role') == 'tool']
    assert tool_msgs_r2 and '红灯' in tool_msgs_r2[0]['content'], \
        f"第 2 轮应包含含红灯信息的工具结果: {tool_msgs_r2[:1]}"

    # 最终回复
    assert reply == '前方检测到车辆和红灯，请保持等待。', f"回复异常: {reply}"

    # 前端编排链路依赖的有界摘要必须与真实执行一致
    orchestration = app_obj.agent.last_orchestration()
    stage_names = [stage['stage'] for stage in orchestration['stages']]
    assert stage_names == [
        'routing', 'prefetch', 'specialists', 'llm', 'tool_loop', 'safety_review',
    ], f"阶段轨迹异常: {stage_names}"
    assert orchestration['intent'] == 'environment', \
        f"意图异常: {orchestration['intent']}"
    assert orchestration['lead_role_name'], '主责角色应有中文名'
    assert orchestration['path_reason'], '决策依据必须解释真实走的路径'
    assert orchestration['suggested_tools'], '摘要应带上建议工具'
    assert orchestration['roles'], '摘要应包含角色协作结论'
    assert orchestration['tools'], '摘要应包含工具轨迹'
    assert orchestration['safety']['guard_passed'] is True, '安全复核应通过'

    # 红灯、高风险通行问题必须绕过 LLM，直接返回保守安全结论
    requests_before_safety_check = len(REQUESTS)
    safety_reply = app_obj.agent.chat('现在能过马路吗？')
    assert len(REQUESTS) == requests_before_safety_check, \
        '红灯高风险问题不应继续请求 LLM'
    assert '红灯' in safety_reply and '不要通行' in safety_reply, \
        f"安全回复异常: {safety_reply}"

    safety_orchestration = app_obj.agent.last_orchestration()
    assert safety_orchestration['path'] == 'deterministic_safety', \
        f"通行问题应走确定性安全路径: {safety_orchestration['path']}"
    llm_stage = next(
        stage for stage in safety_orchestration['stages']
        if stage['stage'] == 'llm'
    )
    assert llm_stage['status'] == 'bypassed', \
        f"通行问题不应进入模型推理: {llm_stage}"

    # 合法偏好意图走本地控制路径，不新增 LLM 请求。
    requests_before_preference = len(REQUESTS)
    preference_reply = app_obj.agent.chat('请把语速调到150')
    assert len(REQUESTS) == requests_before_preference, \
        '本地偏好控制不应请求 LLM'
    assert '150' in preference_reply, f"偏好回复异常: {preference_reply}"

    # 偏好真正落地：持久化 + 应用到 TTS
    assert app_obj.profile.get('voice_rate') == 150, '偏好应已持久化'
    assert app_obj.voice.rate == 150, \
        f"语速应应用到 TTS 引擎，实际 {app_obj.voice.rate}"

    # 对话历史只保留干净的问答对
    assert app_obj.agent.chat_history[-1]['content'] == preference_reply

    # 清理测试写入的用户偏好，避免污染真实数据
    if os.path.exists('data/user_profile.json'):
        os.remove('data/user_profile.json')

    server.shutdown()
    print('=' * 60)
    print('工具调用循环端到端测试全部通过！')
    print(f"  Agent 回复: {reply}")
    print('  工具轮次: get_current_detections -> 最终回答')
    print('  偏好链路: 语速150 -> set_user_preference -> TTS')
    print('  偏好生效: voice_rate=150 已持久化并应用到 TTS 引擎')
    print('=' * 60)


if __name__ == '__main__':
    main()
