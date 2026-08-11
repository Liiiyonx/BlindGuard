# -*- coding: utf-8 -*-
"""
BlindGuard 智能导盲系统 V1.0
"""
import os
import time
import logging
import threading
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('BlindGuard')

# ==================== 语音播报（与原系统相同） ====================
speech_lock = threading.Lock()
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 150)
        _engine.setProperty('volume', 1.0)
        # 设置中文语音
        voices = _engine.getProperty('voices')
        for voice in voices:
            if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                _engine.setProperty('voice', voice.id)
                logger.info(f"使用中文语音: {voice.name}")
                break
    return _engine

def speak_async(text):
    logger.info(f"准备播报: {text}")
    def speak_task():
        logger.info(f"播报线程启动: {text}")
        with speech_lock:
            try:
                engine = get_engine()
                engine.stop()
                engine.say(text)
                engine.runAndWait()
                logger.info(f"播报完成: {text}")
            except RuntimeError as e:
                if "run loop already started" in str(e):
                    logger.warning(f"语音循环冲突，重试: {e}")
                    engine.stop()
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                logger.error(f"语音播报异常: {e}")
    threading.Thread(target=speak_task, daemon=True).start()

# ==================== 摄像头 ====================
class Camera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def open(self, device_id=0):
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
    def __init__(self):
        self.app = Flask(__name__)
        self.camera = Camera()
        self.detector = YOLO("best.pt")
        logger.info(f"模型加载: {self.detector.names}")

        # 类别映射
        self.class_names = {
            0: ('car', '汽车'), 1: ('bicycle', '自行车'),
            2: ('manhole_cover', '井盖'), 3: ('public_facility', '公共设施'),
            4: ('tactile_paving', '盲道'), 5: ('traffic_light', '红绿灯')
        }

        # 状态
        self.running = False
        self.camera_active = False
        self.video_cap = None
        self.video_speed = 1.0
        self.video_paused = False
        self.detections = []
        self.fps = 0
        self.last_speak_time = 0

        self._register_routes()
        logger.info("系统初始化完成")

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

    def _index(self):
        return render_template('index.html')

    def _video_feed(self):
        return Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _stream(self):
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_count = 0
        last_time = time.time()

        while True:
            frame = None

            # 视频模式
            if self.video_cap and not self.video_paused:
                ret, frame = self.video_cap.read()
                if not ret:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_cap.read()

            # 摄像头模式
            if frame is None and self.camera_active:
                frame = self.camera.read()

            if frame is None:
                _, jpg = cv2.imencode('.jpg', black)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
                time.sleep(0.05)
                continue

            # 检测（置信度0.5，与原项目相同）
            results = self.detector(frame, conf=0.5, verbose=False)
            self.detections = []
            logger.info(f"检测到 {len(results[0].boxes) if results and len(results) > 0 else 0} 个目标")

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    if cls_id in self.class_names:
                        en_name, cn_name = self.class_names[cls_id]
                    else:
                        en_name, cn_name = f'class_{cls_id}', f'类别{cls_id}'

                    # 风险等级
                    if en_name in ['car', 'bicycle', 'truck', 'bus']:
                        risk = 'high'
                        color = (0, 0, 255)
                    elif en_name in ['manhole_cover', 'tactile_paving', 'traffic_light']:
                        risk = 'medium'
                        color = (0, 165, 255)
                    else:
                        risk = 'low'
                        color = (0, 255, 0)

                    self.detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_name': en_name,
                        'class_name_cn': cn_name,
                        'confidence': conf,
                        'risk_level': risk
                    })

                    # 绘制
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cn_name} {conf:.0%}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 语音播报
            if self.detections:
                now = time.time()
                logger.info(f"检测到 {len(self.detections)} 个目标，距上次播报: {now - self.last_speak_time:.1f}秒")
                if now - self.last_speak_time >= 3.0:
                    best = max(self.detections, key=lambda x: x['confidence'])
                    msg = f"前方有{best['class_name_cn']}"
                    if best['risk_level'] == 'high':
                        msg = f"危险！{msg}"
                    elif best['risk_level'] == 'medium':
                        msg = f"注意！{msg}"
                    self.last_speak_time = now
                    logger.info(f"触发播报: {msg}")
                    speak_async(msg)

            # FPS
            frame_count += 1
            if time.time() - last_time >= 1:
                self.fps = frame_count
                frame_count = 0
                last_time = time.time()

            cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
            time.sleep(1 / (30 * self.video_speed if self.video_cap else 30))

    # ==================== API ====================
    def _start(self):
        self.running = True
        speak_async("智能导盲系统已启动")
        return jsonify({'success': True, 'message': '系统启动成功'})

    def _stop(self):
        self.running = False
        self._stop_camera()
        self._stop_video()
        speak_async("智能导盲系统已停止")
        return jsonify({'success': True, 'message': '系统已停止'})

    def _status(self):
        pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.video_cap else 0
        total = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_cap else 0
        return jsonify({
            'running': self.running,
            'camera_active': self.camera_active,
            'video_active': self.video_cap is not None,
            'video_paused': self.video_paused,
            'video_position': pos,
            'video_total': total,
            'video_speed': self.video_speed,
            'fps': self.fps,
            'detection_count': len(self.detections)
        })

    def _cam_start(self):
        if not self.running:
            return jsonify({'success': False, 'message': '请先启动系统'})
        if self.camera_active:
            return jsonify({'success': True, 'message': '摄像头已在运行'})
        self._stop_video()
        if self.camera.open():
            self.camera_active = True
            speak_async("摄像头已开启")
            return jsonify({'success': True, 'message': '摄像头启动成功'})
        return jsonify({'success': False, 'message': '无法打开摄像头'})

    def _cam_stop(self):
        self._stop_camera()
        speak_async("摄像头已关闭")
        return jsonify({'success': True, 'message': '摄像头已关闭'})

    def _stop_camera(self):
        self.camera.release()
        self.camera_active = False

    def _upload(self):
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': '没有视频文件'})
        file = request.files['video']
        os.makedirs('uploads', exist_ok=True)
        path = os.path.join('uploads', 'video.mp4')
        file.save(path)
        self._stop_camera()
        self._stop_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': '无法打开视频'})
        self.video_cap = cap
        self.video_speed = 1.0
        self.video_paused = False
        self.running = True
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频已加载: {path}, 总帧数: {total}")
        speak_async("视频已加载，开始检测")
        return jsonify({'success': True, 'message': '视频上传成功', 'total_frames': total})

    def _video_stop(self):
        self._stop_video()
        speak_async("视频播放已停止")
        return jsonify({'success': True, 'message': '视频已停止'})

    def _stop_video(self):
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_paused = False

    def _seek(self):
        data = request.get_json()
        frame = data.get('frame', 0)
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            return jsonify({'success': True})
        return jsonify({'success': False})

    def _speed(self):
        data = request.get_json()
        speed = data.get('speed', 1.0)
        self.video_speed = max(0.25, min(4.0, speed))
        speak_async(f"播放倍速{self.video_speed}倍")
        return jsonify({'success': True, 'speed': self.video_speed})

    def _pause(self):
        self.video_paused = True
        speak_async("视频已暂停")
        return jsonify({'success': True})

    def _resume(self):
        self.video_paused = False
        speak_async("视频继续播放")
        return jsonify({'success': True})

    def _get_detections(self):
        return jsonify({'detections': self.detections})

    def run(self):
        self.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("=" * 50)
    print("BlindGuard 智能导盲系统 V1.0")
    print("=" * 50)
    print("正在初始化...")
    app = BlindGuardApp()
    print("系统就绪！")
    print("访问: http://localhost:5000")
    print("=" * 50)
    app.run()
