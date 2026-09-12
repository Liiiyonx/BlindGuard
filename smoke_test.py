# -*- coding: utf-8 -*-
"""BlindGuard 冒烟测试：不启动服务器，验证应用能完整初始化和响应关键接口"""
import sys
import os

print("=" * 60)
print("BlindGuard 冒烟测试")
print("=" * 60)

# 1. 模块导入
print("\n[1/5] 导入核心模块...")
try:
    from core.agent import BlindGuardAgent, load_agent_config
    print("  [OK] core.agent")
    from core.config_manager import ConfigManager
    print("  [OK] core.config_manager")
    from core.detection_engine import DetectionEngine
    print("  [OK] core.detection_engine")
    from core.risk_evaluator import RiskEvaluator
    print("  [OK] core.risk_evaluator")
    from core.scene_analyzer import SceneAnalyzer
    print("  [OK] core.scene_analyzer")
    from core.voice_announcer import VoiceAnnouncer
    print("  [OK] core.voice_announcer")
except Exception as e:
    print(f"  [FAIL] 模块导入失败: {e}")
    sys.exit(1)

# 2. Agent 独立功能测试
print("\n[2/5] 测试 Agent 播报与对话（无 LLM 回退路径）...")
try:
    agent = BlindGuardAgent(load_agent_config('config.yaml'))
    fake_dets = [
        {'bbox': [10, 10, 200, 300], 'class_name': 'car', 'class_name_cn': '汽车',
         'confidence': 0.87, 'risk_level': 'high'},
        {'bbox': [400, 50, 500, 150], 'class_name': 'tactile_paving', 'class_name_cn': '盲道',
         'confidence': 0.65, 'risk_level': 'medium'},
    ]
    agent.update_memory(fake_dets, 'high')
    ann = agent.generate_announcement(fake_dets, 'high')
    print(f"  [OK] 播报: {ann}")
    reply = agent.chat("前面有什么危险？", fake_dets, 'high')
    print(f"  [OK] 对话: {reply}")
    st = agent.status()
    print(f"  [OK] Agent 状态: llm_active={st['llm_active']} model={st['model']}")
except Exception as e:
    print(f"  [FAIL] Agent 测试失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 3. 主应用初始化（加载 YOLO 模型）
print("\n[3/5] 初始化主应用（加载 best.pt 模型，需要几秒）...")
try:
    from app import BlindGuardApp
    app_obj = BlindGuardApp()
    names = app_obj.detector.get_class_names()
    print(f"  [OK] 应用初始化完成，模型类别: {list(names.values())}")
    print(f"  [OK] 核心模块接入: detector={type(app_obj.detector).__name__} "
          f"risk={type(app_obj.risk).__name__} scene={type(app_obj.scene).__name__} "
          f"voice={type(app_obj.voice).__name__}")
    print(f"  [OK] Agent 接入: llm_ok={app_obj.agent.llm_ok}")
except Exception as e:
    print(f"  [FAIL] 应用初始化失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 4. Flask 接口测试
print("\n[4/5] 测试关键 API 接口...")
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
    print(f"  [OK] GET /api/agent/status -> enabled={data['enabled']} llm_active={data['llm_active']}")

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

# 5. 模型文件检查
print("\n[5/5] 模型文件检查...")
for f in ['best.pt', 'yolov10n.pt']:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  [OK] {f} ({size:.1f}MB)")
    else:
        print(f"  [!!] {f} 不存在")

print("\n" + "=" * 60)
print("冒烟测试全部通过！系统可以启动演示。")
print("启动命令: python run.py 或 python app.py")
print("=" * 60)
