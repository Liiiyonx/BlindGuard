# -*- coding: utf-8 -*-
"""
BlindGuard 智能导盲系统 V2.1

架构：检测引擎 → 红绿灯状态/目标跟踪 → 风险评估器 → 场景分析器 → 智能体(LLM+工具) → 语音播报器
各层由 core/ 下独立模块实现，本文件只负责装配与调度。

性能模型：检测在独立管线线程中单实例运行，多浏览器客户端共享同一份
最新标注帧（MJPEG 接口只做转发），FPS 与客户端数量无关。
"""
import os
import time
import logging
import threading
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np

from core.config_manager import ConfigManager
from core.detection_engine import DetectionEngine
from core.risk_evaluator import RiskEvaluator
from core.scene_analyzer import SceneAnalyzer
from core.voice_announcer import VoiceAnnouncer
from core.agent import BlindGuardAgent
from core.tracker import SimpleTracker
from core.traffic_light import TrafficLightClassifier, STATE_CN
from core.user_profile import UserProfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('BlindGuard')

# 风险等级 -> 绘制颜色 (BGR)
RISK_COLORS = {
    'critical': (0, 0, 255),      # 红
    'high': (0, 69, 255),         # 橙红
    'medium': (0, 165, 255),      # 橙
    'low': (255, 200, 0),         # 蓝
    'safe': (0, 255, 0),          # 绿
}

# 风险等级 -> 语音播报优先级
RISK_VOICE_PRIORITY = {
    'critical': VoiceAnnouncer.PRIORITY_CRITICAL,
    'high': VoiceAnnouncer.PRIORITY_HIGH,
    'medium': VoiceAnnouncer.PRIORITY_MEDIUM,
    'low': VoiceAnnouncer.PRIORITY_LOW,
    'safe': VoiceAnnouncer.PRIORITY_LOW,
}

# 红绿灯状态的检测框标签后缀（cv2 不支持中文，用字母缩写）
LIGHT_TAG = {'red': ' [R]', 'yellow': ' [Y]', 'green': ' [G]'}

# 红绿灯类别名（自定义模型与 COCO 命名两种）
TRAFFIC_LIGHT_CLASSES = ('traffic_light', 'traffic light')


# ==================== 摄像头 ====================
class Camera:
    def __init__(self, device_id=0, width=640, height=480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def open(self, device_id=None):
        if device_id is None:
            device_id = self.device_id
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        ret, f = self.cap.read()
        if ret:
            self.frame = f.copy()
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        while self.running and self.cap:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = f.copy()
            time.sleep(1/30)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None


# ==================== 主应用 ====================
class BlindGuardApp:
    def __init__(self, config_path='config.yaml'):
        self.app = Flask(__name__)

        # 配置（统一入口：YAML + .env 环境变量覆盖）
        self.config = ConfigManager(config_path)

        # 摄像头
        cam_cfg = self.config.get_section('camera')
        self.camera = Camera(
            device_id=cam_cfg.get('device_id', 0),
            width=cam_cfg.get('width', 640),
            height=cam_cfg.get('height', 480),
        )

        # 检测引擎（主模型：通用交通类）
        model_cfg = self.config.get_section('model')
        self.detector = DetectionEngine(
            model_path=model_cfg.get('path', 'best.pt'),
            confidence_threshold=model_cfg.get('confidence_threshold', 0.5),
            iou_threshold=model_cfg.get('iou_threshold', 0.45),
            device=model_cfg.get('device', ''),
            img_size=model_cfg.get('img_size', 640),
        )
        logger.info(f"主模型加载: {self.detector.get_class_names()}")

        # 主模型类别过滤（COCO 80 类 -> 导盲相关子集）
        focus_names = model_cfg.get('focus_classes') or []
        name2id = {v: k for k, v in self.detector.get_class_names().items()}
        self._focus_class_ids = [name2id[n] for n in focus_names if n in name2id] or None

        # 辅助检测引擎（定制类专家：井盖/盲道/公共设施），未配置则单模型
        self.aux_detector = None
        self._aux_class_ids = None
        aux_path = model_cfg.get('aux_model_path') or ''
        if aux_path:
            try:
                self.aux_detector = DetectionEngine(
                    model_path=aux_path,
                    confidence_threshold=model_cfg.get('confidence_threshold', 0.5),
                    iou_threshold=model_cfg.get('iou_threshold', 0.45),
                    device=model_cfg.get('device', ''),
                    img_size=model_cfg.get('aux_img_size', 640),
                )
                aux_names = model_cfg.get('aux_classes') or []
                aux_name2id = {v: k for k, v in self.aux_detector.get_class_names().items()}
                self._aux_class_ids = [aux_name2id[n] for n in aux_names
                                       if n in aux_name2id] or None
                logger.info(f"辅助模型加载: {aux_path} (定制类白名单: {aux_names})")
            except Exception as e:
                logger.warning(f"辅助模型加载失败，退回单模型: {e}")
                self.aux_detector = None

        # 类别中英文映射（唯一来源：config.yaml class_mapping，默认表兜底）
        self.class_mapping = self.config.get('class_mapping', {}) or {}

        # 风险评估器（yaml 的 area_thresholds 对应 RiskEvaluator 的 distance 阈值）
        risk_cfg = self.config.get_section('risk')
        self.risk = RiskEvaluator(
            risk_thresholds={'distance': risk_cfg.get('thresholds', {})},
            high_risk_classes=risk_cfg.get('high_risk_classes'),
            medium_risk_classes=risk_cfg.get('medium_risk_classes'),
            low_risk_classes=risk_cfg.get('low_risk_classes'),
            class_mapping=self.class_mapping,
        )

        # 场景分析器 / 目标跟踪器 / 红绿灯状态识别
        self.scene = SceneAnalyzer(class_mapping=self.class_mapping)
        self.tracker = SimpleTracker()
        self.light_classifier = TrafficLightClassifier()

        # 语音播报器
        voice_cfg = self.config.get_section('voice')
        self.voice = VoiceAnnouncer(
            rate=voice_cfg.get('rate', 180),
            volume=voice_cfg.get('volume', 1.0),
            cooldown=voice_cfg.get('cooldown', 3.0),
        )
        self.voice.start()

        # 用户偏好（跨重启持久化），启动时应用已保存的语速
        self.profile = UserProfile()
        saved_rate = self.profile.get('voice_rate')
        if saved_rate:
            self.voice.set_rate(int(saved_rate))

        # 智能体（LLM 推理 + 工具调用 + 时序记忆 + 对话）
        self.agent = BlindGuardAgent(self.config.get_section('agent'),
                                     class_mapping=self.class_mapping)
        self.agent.set_context(self)

        # 状态
        self.running = False
        self.camera_active = False
        self.video_cap = None
        self.video_speed = 1.0
        self.video_paused = False
        self.detections = []
        self.scene_info = {}
        self.fps = 0
        self.last_speak_time = 0

        self.detections_lock = threading.Lock()
        # 视频句柄锁（管线线程 read/seek，HTTP 线程 status/seek）
        self.video_lock = threading.Lock()
        # 最新媒体帧（标注后 JPEG 供视频流转发；原始帧供 VLM 看图）
        self.media_lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_raw = None
        self._frame_size = (640, 480)

        # 播报生成并发控制，避免 LLM 调用堆积
        self._generating = False
        # 最近一条播报，供前端展示
        self.last_message = "系统就绪，请启动系统开始检测"
        self.overall_risk = 'safe'
        # 播报事件时间线（Frigate 式事件流：供前端时间线面板回看）
        self.events = deque(maxlen=50)
        self.events.append({'time': time.strftime('%H:%M:%S'),
                            'text': '系统就绪', 'level': 'safe'})
        # 最近一次播报的端到端延迟（检测帧完成 -> 播报文本生成）
        self.last_announce_latency_ms = None

        self._register_routes()
        # 单实例检测管线线程：随进程常驻，空闲时低成本休眠
        threading.Thread(target=self._pipeline_loop, daemon=True,
                         name='DetectionPipeline').start()
        logger.info("系统初始化完成")
        if self.agent.llm_ok:
            logger.info(f"智能体已接入 LLM: {self.agent.model}，工具调用已就绪")
        else:
            logger.info("智能体未接入 LLM，使用模板播报 + 本地对话（在 .env 配置 BLINDGUARD_AGENT_LLM_API_KEY 启用）")
        if self.agent.vlm_ok:
            logger.info(f"多模态看图已接入: {self.agent.vision_model}")
        else:
            logger.info("多模态看图未配置（config.yaml agent.vision），看图工具将提示未启用")

    def _register_routes(self):
        self.app.add_url_rule('/', 'index', self._index)
        self.app.add_url_rule('/video_feed', 'video_feed', self._video_feed)
        self.app.add_url_rule('/api/system/start', 'start', self._start, methods=['POST'])
        self.app.add_url_rule('/api/system/stop', 'stop', self._stop, methods=['POST'])
        self.app.add_url_rule('/api/system/status', 'status', self._status)
        self.app.add_url_rule('/api/camera/start', 'cam_start', self._cam_start, methods=['POST'])
        self.app.add_url_rule('/api/camera/stop', 'cam_stop', self._cam_stop, methods=['POST'])
        self.app.add_url_rule('/api/video/upload', 'upload', self._upload, methods=['POST'])
        self.app.add_url_rule('/api/video/stop', 'video_stop', self._video_stop, methods=['POST'])
        self.app.add_url_rule('/api/video/seek', 'seek', self._seek, methods=['POST'])
        self.app.add_url_rule('/api/video/speed', 'speed', self._speed, methods=['POST'])
        self.app.add_url_rule('/api/video/pause', 'pause', self._pause, methods=['POST'])
        self.app.add_url_rule('/api/video/resume', 'resume', self._resume, methods=['POST'])
        self.app.add_url_rule('/api/detections', 'detections', self._get_detections)
        self.app.add_url_rule('/api/chat', 'chat', self._chat, methods=['POST'])
        self.app.add_url_rule('/api/agent/status', 'agent_status', self._agent_status)
        # PWA
        self.app.add_url_rule('/manifest.json', 'manifest',
                              lambda: send_from_directory('static', 'manifest.json'))
        self.app.add_url_rule('/sw.js', 'sw',
                              lambda: send_from_directory('static', 'sw.js',
                                                          mimetype='application/javascript'))
        self.app.add_url_rule('/icon_192.png', 'icon192',
                              lambda: send_from_directory('static', 'icon_192.png'))
        self.app.add_url_rule('/icon_512.png', 'icon512',
                              lambda: send_from_directory('static', 'icon_512.png'))

    def _index(self):
        return render_template('index.html')

    # ==================== Agent 上下文适配（工具数据源） ====================
    def get_detections(self):
        with self.detections_lock:
            return list(self.detections)

    def get_overall_risk(self):
        return self.overall_risk

    def get_scene(self):
        """瘦身的场景信息（剔除重量级 detections 列表）"""
        scene = self.scene_info
        return {k: scene.get(k) for k in
                ('scene_type', 'scene_type_cn', 'summary', 'object_count',
                 'movement_analysis', 'spatial_analysis', 'object_distribution')
                if k in scene}

    def get_frame_jpeg(self):
        """当前原始画面 JPEG（供 VLM 看图），无画面返回 None"""
        with self.media_lock:
            raw = self._latest_raw
        if raw is None:
            return None
        ok, jpg = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes() if ok else None

    def get_preferences(self):
        return self.profile.get_all()

    def set_preference(self, key, value):
        ok, message = self.profile.set(key, value)
        if ok and key == 'voice_rate':
            try:
                self.voice.set_rate(int(float(value)))
            except (TypeError, ValueError):
                pass
        return {'success': ok, 'message': message}

    # ==================== 检测管线（单实例线程） ====================
    def _pipeline_loop(self):
        """检测 → 红绿灯 → 跟踪 → 风险 → 场景 → 绘制 → 共享帧，独立于客户端数量"""
        frame_count = 0
        last_fps_time = time.time()

        while True:
            if not self.running:
                with self.media_lock:
                    self._latest_jpeg = None
                    self._latest_raw = None
                time.sleep(0.1)
                continue

            frame = None

            # 视频模式
            with self.video_lock:
                if self.video_cap and not self.video_paused:
                    ret, frame = self.video_cap.read()
                    if not ret and self.video_cap:
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.video_cap.read()

            # 摄像头模式
            if frame is None and self.camera_active:
                frame = self.camera.read()

            if frame is None:
                # 无新帧（如视频暂停）：保留最后一帧画面，短暂休眠
                time.sleep(0.05)
                continue

            t0 = time.time()
            frame_h, frame_w = frame.shape[:2]
            self._frame_size = (frame_w, frame_h)

            # --- 管线：检测(主+辅) → 红绿灯状态 → 跟踪 → 风险 → 场景 ---
            detections = self.detector.detect(frame, classes=self._focus_class_ids)
            if self.aux_detector is not None:
                aux_dets = self.aux_detector.detect(frame, classes=self._aux_class_ids)
                detections = self._merge_detections(detections, aux_dets)
            for det in detections:
                if det.get('class_name') in TRAFFIC_LIGHT_CLASSES:
                    det['light_state'] = self.light_classifier.classify(
                        frame, det.get('bbox'))['state']
            self.tracker.update(detections)

            risk_results = self.risk.evaluate(detections, frame_width=frame_w)
            enriched = self._enrich_detections(risk_results)
            overall = self.risk.get_overall_risk(risk_results)
            self.scene_info = self.scene.analyze(frame, detections)

            with self.detections_lock:
                self.detections = enriched
            self.overall_risk = overall['overall_level']

            # 智能体：每帧更新时序记忆（不含 LLM，开销小）
            self.agent.update_memory(enriched, self.overall_risk,
                                     frame_w, frame_h, scene=self.get_scene())

            # 绘制检测框（叠加红绿灯状态与接近趋势）
            for det in enriched:
                x1, y1, x2, y2 = det['bbox']
                color = RISK_COLORS.get(det['risk_level'], (0, 255, 0))
                tag = LIGHT_TAG.get(det.get('light_state'), '')
                arrow = ''
                if det.get('trend') == 'approaching':
                    arrow = ' >>'
                elif det.get('trend') == 'receding':
                    arrow = ' <<'
                label = f"{det['class_name']}{tag}{arrow} {det['confidence']:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 发布最新帧（原始帧与标注帧共享引用即可，下一帧会整体替换）
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.media_lock:
                self._latest_jpeg = jpg.tobytes()
                self._latest_raw = frame

            # 语音播报（分级保守策略过滤后触发，异步生成不阻塞管线）
            announce_dets = self.agent.should_announce(enriched)
            if announce_dets and not self._generating:
                now = time.time()
                if now - self.last_speak_time >= self.agent.announce_cooldown:
                    self.last_speak_time = now
                    with self.video_lock:
                        frame_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) \
                            if self.video_cap else 0
                    threading.Thread(
                        target=self._async_announce,
                        args=(list(announce_dets), self.overall_risk,
                              frame_w, frame_h, time.time(), frame_pos),
                        daemon=True
                    ).start()

            # FPS
            frame_count += 1
            if time.time() - last_fps_time >= 1:
                self.fps = frame_count
                frame_count = 0
                last_fps_time = time.time()

            # 节奏：视频按倍速、摄像头 30fps；实际推理更慢时不额外等待
            target_interval = 1 / (30 * self.video_speed) if self.video_cap else 1 / 30
            delay = target_interval - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)

    def _stream(self):
        """MJPEG 视频流：只转发管线产出的最新帧，与客户端数量无关"""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        _, black_jpg = cv2.imencode('.jpg', black)
        while True:
            with self.media_lock:
                payload = self._latest_jpeg
            if payload is None:
                payload = black_jpg.tobytes()
                time.sleep(0.1)
            else:
                time.sleep(1 / 25)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + payload + b'\r\n'

    def _video_feed(self):
        return Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @staticmethod
    def _box_iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_detections(self, main_dets, aux_dets):
        """双模型合并：主模型优先，辅助模型仅补充与其不重叠的同类别框
        （辅助模型经类别白名单过滤，旧模型的 person→car 误检在此被拦截）"""
        merged = list(main_dets)
        for a in aux_dets:
            duplicate = any(
                m['class_name'] == a['class_name']
                and self._box_iou(m['bbox'], a['bbox']) > 0.55
                for m in main_dets
            )
            if not duplicate:
                merged.append(a)
        return merged

    def _enrich_detections(self, risk_results):
        """为风险评估结果补充中文名等前端所需字段"""
        enriched = []
        for r in risk_results:
            det = dict(r)
            det['class_name_cn'] = self.class_mapping.get(
                det['class_name'], det['class_name'])
            if det.get('light_state'):
                det['light_state_cn'] = STATE_CN.get(det['light_state'],
                                                     det['light_state'])
            enriched.append(det)
        return enriched

    # ==================== API ====================
    def _start(self):
        self.running = True
        self.voice.speak_system("智能导盲系统已启动")
        return jsonify({'success': True, 'message': '系统启动成功'})

    def _stop(self):
        self.running = False
        self._stop_camera()
        self._stop_video()
        self.voice.speak_system("智能导盲系统已停止")
        return jsonify({'success': True, 'message': '系统已停止'})

    def _status(self):
        with self.video_lock:
            pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.video_cap else 0
            total = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_cap else 0
        with self.detections_lock:
            det_count = len(self.detections)
        return jsonify({
            'running': self.running,
            'camera_active': self.camera_active,
            'video_active': self.video_cap is not None,
            'video_paused': self.video_paused,
            'video_position': pos,
            'video_total': total,
            'video_speed': self.video_speed,
            'fps': self.fps,
            'detection_count': det_count,
            'overall_risk': self.overall_risk,
            'scene_type': self.scene_info.get('scene_type', ''),
            'scene_type_cn': self.scene_info.get('scene_type_cn', ''),
            'last_message': self.last_message,
            'events': list(self.events)[-8:][::-1],
            'frame_w': self._frame_size[0],
            'frame_h': self._frame_size[1],
        })

    def _cam_start(self):
        if not self.running:
            return jsonify({'success': False, 'message': '请先启动系统'})
        if self.camera_active:
            return jsonify({'success': True, 'message': '摄像头已在运行'})
        self._stop_video()
        if self.camera.open():
            self.camera_active = True
            self.voice.speak_system("摄像头已开启")
            return jsonify({'success': True, 'message': '摄像头启动成功'})
        return jsonify({'success': False, 'message': '无法打开摄像头'})

    def _cam_stop(self):
        self._stop_camera()
        self.voice.speak_system("摄像头已关闭")
        return jsonify({'success': True, 'message': '摄像头已关闭'})

    def _stop_camera(self):
        self.camera.release()
        self.camera_active = False

    def _upload(self):
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': '没有视频文件'})
        file = request.files['video']
        if not file.filename:
            return jsonify({'success': False, 'message': '没有视频文件'})
        os.makedirs('uploads', exist_ok=True)
        path = os.path.join('uploads', 'video.mp4')
        file.save(path)
        self._stop_camera()
        self._stop_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': '无法打开视频'})
        self.tracker.reset()
        self.video_cap = cap
        self.video_speed = 1.0
        self.video_paused = False
        self.running = True
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频已加载: {path}, 总帧数: {total}")
        self.voice.speak_system("视频已加载，开始检测")
        return jsonify({'success': True, 'message': '视频上传成功', 'total_frames': total})

    def _video_stop(self):
        self._stop_video()
        self.voice.speak_system("视频播放已停止")
        return jsonify({'success': True, 'message': '视频已停止'})

    def _stop_video(self):
        with self.video_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            self.video_paused = False

    def _seek(self):
        data = request.get_json(silent=True) or {}
        frame = data.get('frame', 0)
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'message': '帧号无效'})
        with self.video_lock:
            if self.video_cap and frame >= 0:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                return jsonify({'success': True})
        return jsonify({'success': False})

    def _speed(self):
        data = request.get_json(silent=True) or {}
        speed = data.get('speed', 1.0)
        self.video_speed = max(0.25, min(4.0, speed))
        self.voice.speak_system(f"播放倍速{self.video_speed}倍")
        return jsonify({'success': True, 'speed': self.video_speed})

    def _pause(self):
        self.video_paused = True
        self.voice.speak_system("视频已暂停")
        return jsonify({'success': True})

    def _resume(self):
        self.video_paused = False
        self.voice.speak_system("视频继续播放")
        return jsonify({'success': True})

    def _get_detections(self):
        with self.detections_lock:
            return jsonify({'detections': list(self.detections)})

    # ==================== 智能体相关 ====================
    def _async_announce(self, dets, risk, frame_w, frame_h, t_frame_done=None, frame_pos=0):
        """后台线程：调用智能体生成播报，完成后送入语音队列，并统计端到端延迟"""
        try:
            text = self.agent.generate_announcement(dets, risk, frame_w, frame_h)
            if text:
                if t_frame_done is not None:
                    self.last_announce_latency_ms = round((time.time() - t_frame_done) * 1000)
                    logger.info(f"播报端到端延迟: {self.last_announce_latency_ms}ms "
                                f"(LLM={'在线' if self.agent.llm_ok else '离线模板'})")
                logger.info(f"智能体播报: {text}")
                self.last_message = text
                self.events.append({'time': time.strftime('%H:%M:%S'),
                                    'text': text, 'level': risk, 'frame': frame_pos})
                priority = RISK_VOICE_PRIORITY.get(risk, VoiceAnnouncer.PRIORITY_MEDIUM)
                self.voice.speak(text, priority=priority)
        except Exception as e:
            logger.error(f"智能体播报异常: {e}")

    def _chat(self):
        data = request.get_json(silent=True) or {}
        question = (data.get('message') or data.get('q') or '').strip()
        if not question:
            return jsonify({'success': False, 'message': '请输入问题'})
        with self.detections_lock:
            dets = list(self.detections)
        risk = self.overall_risk
        frame_w, frame_h = self._frame_size
        try:
            reply = self.agent.chat(question, dets, risk, frame_w, frame_h)
            logger.info(f"用户问: {question} | Agent答: {reply}")
            return jsonify({
                'success': True,
                'reply': reply,
                'llm_active': self.agent.llm_ok,
                'detection_count': len(dets)
            })
        except Exception as e:
            logger.error(f"对话异常: {e}")
            return jsonify({'success': False, 'message': f'对话失败: {e}'})

    def _agent_status(self):
        status = self.agent.status()
        if self.last_announce_latency_ms is not None:
            status['announce_latency_ms'] = self.last_announce_latency_ms
        return jsonify(status)

    def run(self):
        server_cfg = self.config.get_section('server')
        host = server_cfg.get('host', '0.0.0.0')
        port = server_cfg.get('port', 5000)
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        finally:
            self.voice.stop()
            self._stop_camera()
            self._stop_video()


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("=" * 50)
    print("BlindGuard 智能导盲系统 V2.1")
    print("=" * 50)
    print("正在初始化...")
    app = BlindGuardApp()
    print("系统就绪！")
    print("访问: http://localhost:5000")
    print("=" * 50)
    app.run()
