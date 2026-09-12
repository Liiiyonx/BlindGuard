# -*- coding: utf-8 -*-
"""
工具调用循环端到端测试：
用本地模拟的 OpenAI 兼容端点验证 Agent 的"感知-推理-行动"闭环，
无需真实 API Key 即可确定性测试 function calling 链路与偏好生效。

模拟脚本：
  第 1 轮: LLM 请求调用 get_current_detections 工具
  第 2 轮: LLM 请求调用 set_user_preference 工具 (voice_rate=150)
  第 3 轮: LLM 给出最终回答
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
        elif n == 2:
            msg = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_2', 'type': 'function',
                 'function': {'name': 'set_user_preference',
                              'arguments': json.dumps({'key': 'voice_rate',
                                                       'value': '150'})}}]}
        else:
            msg = {'role': 'assistant',
                   'content': '红灯亮起，请等待绿灯再通行。'}
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

    reply = app_obj.agent.chat('现在能过马路吗？')

    # 断言三轮请求与工具声明
    assert len(REQUESTS) == 3, f"期望 3 轮请求，实际 {len(REQUESTS)}"
    tools = REQUESTS[0].get('tools') or []
    assert len(tools) == 6, f"第 1 轮应声明 6 个工具，实际 {len(tools)}"
    names = [t['function']['name'] for t in tools]
    assert 'look_at_frame' in names, '应包含看图工具'

    # 第 2 轮消息中应包含第 1 轮的工具结果（检测列表，含红灯）
    tool_msgs_r2 = [m for m in REQUESTS[1]['messages'] if m.get('role') == 'tool']
    assert tool_msgs_r2 and '红灯' in tool_msgs_r2[0]['content'], \
        f"第 2 轮应包含含红灯信息的工具结果: {tool_msgs_r2[:1]}"

    # 第 3 轮消息中应包含第 2 轮的偏好设置结果（按 tool_call_id 对应）
    tool_msgs_r3 = {m.get('tool_call_id'): m.get('content', '')
                    for m in REQUESTS[2]['messages'] if m.get('role') == 'tool'}
    assert '150' in tool_msgs_r3.get('call_2', ''), \
        f"第 3 轮应包含偏好设置结果: {tool_msgs_r3}"

    # 最终回复
    assert reply == '红灯亮起，请等待绿灯再通行。', f"回复异常: {reply}"

    # 偏好真正落地：持久化 + 应用到 TTS
    assert app_obj.profile.get('voice_rate') == 150, '偏好应已持久化'
    assert app_obj.voice.rate == 150, \
        f"语速应应用到 TTS 引擎，实际 {app_obj.voice.rate}"

    # 对话历史只保留干净的问答对
    assert app_obj.agent.chat_history[-1]['content'] == reply

    # 清理测试写入的用户偏好，避免污染真实数据
    if os.path.exists('data/user_profile.json'):
        os.remove('data/user_profile.json')

    server.shutdown()
    print('=' * 60)
    print('工具调用循环端到端测试全部通过！')
    print(f"  Agent 回复: {reply}")
    print('  工具轮次: get_current_detections -> set_user_preference -> 最终回答')
    print('  偏好生效: voice_rate=150 已持久化并应用到 TTS 引擎')
    print('=' * 60)


if __name__ == '__main__':
    main()
