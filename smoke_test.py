# -*- coding: utf-8 -*-
"""BlindGuard 冒烟测试：不启动服务器，验证应用能完整初始化和响应关键接口"""
import sys
import os

print("=" * 60)
print("BlindGuard 冒烟测试 (V2.1)")
print("=" * 60)

# 1. 模块导入
print("\n[1/7] 导入核心模块...")
try:
    from core.agent import BlindGuardAgent, load_agent_config, AGENT_TOOLS
    from core.config_manager import ConfigManager
    from core.detection_engine import DetectionEngine
    from core.risk_evaluator import RiskEvaluator
    from core.scene_analyzer import SceneAnalyzer
    from core.voice_announcer import VoiceAnnouncer
    from core.tracker import SimpleTracker
    from core.traffic_light import TrafficLightClassifier
    from core.user_profile import UserProfile
    from core.labels import get_class_cn
    print("  [OK] core 全部 9 个模块")
except Exception as e:
    print(f"  [FAIL] 模块导入失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 2. Agent 独立功能测试（无 LLM 回退路径）
print("\n[2/7] 测试 Agent 播报/对话/工具注册（无 LLM 回退路径）...")
try:
    agent = BlindGuardAgent(load_agent_config('config.yaml'))
    fake_dets = [
        {'bbox': [10, 10, 200, 300], 'class_name': 'car', 'class_name_cn': '汽车',
         'confidence': 0.87, 'risk_level': 'high', 'trend': 'approaching',
         'track_id': 1, 'track_age': 2.5},
        {'bbox': [400, 50, 500, 150], 'class_name': 'tactile_paving', 'class_name_cn': '盲道',
         'confidence': 0.65, 'risk_level': 'medium'},
    ]
    agent.update_memory(fake_dets, 'high', 640, 480)
    ann = agent.generate_announcement(fake_dets, 'high')
    print(f"  [OK] 播报: {ann}")
    reply = agent.chat("前面有什么危险？", fake_dets, 'high')
    print(f"  [OK] 对话: {reply}")
    # 红灯兜底播报
    red_dets = [{'bbox': [250, 0, 390, 100], 'class_name': 'traffic_light',
                 'confidence': 0.9, 'risk_level': 'high', 'light_state': 'red'}]
    ann_red = agent.generate_announcement(red_dets, 'high')
    print(f"  [OK] 红灯播报: {ann_red}")
    assert ann and reply and '红灯' in ann_red
    print(f"  [OK] 工具注册: {len(AGENT_TOOLS)} 个")
    st = agent.status()
    print(f"  [OK] Agent 状态: llm_active={st['llm_active']} tools_supported={st['tools_supported']} "
          f"vision_active={st['vision_active']}")
except Exception as e:
    print(f"  [FAIL] Agent 测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 3. 目标跟踪
print("\n[3/7] 测试目标跟踪（面积增大 → 接近趋势）...")
try:
    tracker = SimpleTracker(trend_window=0.0, trend_ratio=1.25)
    d1 = [{'bbox': [0, 0, 100, 100], 'class_name': 'car', 'area': 10000}]
    tracker.update(d1)
    # 框扩大且保持高 IoU（IoU≈0.44 > 0.3 匹配阈值），面积增至 2.25 倍 → 接近
    d2 = [{'bbox': [0, 0, 150, 150], 'class_name': 'car', 'area': 22500}]
    tracker.update(d2)
    assert d2[0]['track_id'] == d1[0]['track_id'], "同一目标 track_id 应稳定"
    assert d2[0]['trend'] == 'approaching', f"应为 approaching，实际 {d2[0]['trend']}"
    print(f"  [OK] track_id={d2[0]['track_id']} 稳定, trend={d2[0]['trend']}")
except Exception as e:
    print(f"  [FAIL] 跟踪测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 4. 红绿灯状态识别
print("\n[4/7] 测试红绿灯状态识别（合成图像）...")
import numpy as np
import cv2
try:
    clf = TrafficLightClassifier(min_pixels=20)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 50), 30, (0, 0, 255), -1)   # 红色圆 (BGR)
    r = clf.classify(img, [280, 10, 360, 90])
    assert r['state'] == 'red', f"应为 red，实际 {r['state']} (score={r['score']})"
    print(f"  [OK] 红灯识别: state={r['state']} score={r['score']}")
    blank = clf.classify(np.zeros((480, 640, 3), dtype=np.uint8), [280, 10, 360, 90])
    print(f"  [OK] 无灯区域兜底: state={blank['state']}")
except Exception as e:
    print(f"  [FAIL] 红绿灯测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 5. 用户偏好记忆
print("\n[5/7] 测试用户偏好记忆（临时文件）...")
try:
    ppath = 'data/_smoke_profile.json'
    prof = UserProfile(path=ppath)
    ok, msg = prof.set('voice_rate', 200)
    assert ok, msg
    ok2, msg2 = prof.set('voice_rate', 999)
    assert not ok2, "超出范围的语速应被拒绝"
    prof2 = UserProfile(path=ppath)  # 重新加载验证持久化
    assert prof2.get('voice_rate') == 200
    os.remove(ppath)
    print(f"  [OK] 设置: {msg}；非法值已拒绝；持久化正常")
except Exception as e:
    print(f"  [FAIL] 偏好测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 6. 主应用初始化（加载 YOLO 模型）
print("\n[6/7] 初始化主应用（加载 best.pt 模型，需要几秒）...")
try:
    from app import BlindGuardApp
    app_obj = BlindGuardApp()
    names = app_obj.detector.get_class_names()
    print(f"  [OK] 应用初始化完成，模型类别: {list(names.values())}")
    print(f"  [OK] 核心模块接入: detector/risk/scene/tracker/light/profile/agent")
    print(f"  [OK] Agent 接入: llm_ok={app_obj.agent.llm_ok} vlm_ok={app_obj.agent.vlm_ok}")
    # 播报分级保守策略：low 等级与低置信度应被过滤
    sample = [{'risk_level': 'high', 'confidence': 0.9},
              {'risk_level': 'low', 'confidence': 0.9},
              {'risk_level': 'critical', 'confidence': 0.2}]
    kept = app_obj.agent.should_announce(sample)
    assert len(kept) == 1 and kept[0]['risk_level'] == 'high', \
        f"分级策略应只保留 high 一条，实际 {len(kept)} 条"
    print("  [OK] 播报分级策略: low 等级与低置信度目标已过滤")
except Exception as e:
    print(f"  [FAIL] 应用初始化失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 7. Flask 接口测试
print("\n[7/7] 测试关键 API 接口...")
try:
    client = app_obj.app.test_client()

    r = client.get('/')
    assert r.status_code == 200, f"首页返回 {r.status_code}"
    print(f"  [OK] GET / -> 200 (前端页面)")

    r = client.get('/api/system/status')
    data = r.get_json()
    print(f"  [OK] GET /api/system/status -> running={data['running']} fps={data['fps']}")

    r = client.get('/api/agent/status')
    data = r.get_json()
    print(f"  [OK] GET /api/agent/status -> llm_active={data['llm_active']} "
          f"tools={data['tool_count']} vision={data['vision_active']}")

    r = client.post('/api/chat', json={'message': '周围有什么？'})
    data = r.get_json()
    print(f"  [OK] POST /api/chat -> success={data['success']} reply={data.get('reply','')[:30]}...")

    r = client.get('/api/detections')
    data = r.get_json()
    print(f"  [OK] GET /api/detections -> {len(data['detections'])} 个目标")
except Exception as e:
    print(f"  [FAIL] 接口测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("冒烟测试全部通过！系统可以启动演示。")
print("启动命令: python run.py 或 python app.py")
print("=" * 60)
