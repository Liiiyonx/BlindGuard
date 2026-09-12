# -*- coding: utf-8 -*-
"""
BlindGuard 答辩指标评测脚本

四种模式（结果统一追加写入 evaluation/metrics_report.md，答辩可直接引用）：

  python evaluate.py perf [--source uploads/video.mp4] [--frames 300]
      推理性能与播报延迟：管线各阶段耗时、FPS、CPU/内存占用、
      模板播报端到端延迟（LLM 网络耗时另计，通常 1-3 秒）

  python evaluate.py accuracy --data data_eval.yaml
      模型精度：mAP50 / mAP50-95 / 各类别 AP（需 YOLO 格式测试集，
      可先用 bootstrap 模式生成伪标签再人工校正）

  python evaluate.py light [--dir crops_dir] [--synthetic 60]
      红绿灯识别准确率：--dir 指向含 red/green/yellow 子目录的真实裁剪图，
      或 --synthetic N 生成 N 张合成图像自检（无需数据）

  python evaluate.py bootstrap [--source video.mp4] [--stride 30] [--frames 150]
      从视频抽帧 + 模型伪标签输出 YOLO 格式，人工用 labelImg/CVAT 校正后
      即可作为测试集
"""

import os
import sys
import time
import argparse
import statistics
from datetime import datetime

REPORT_PATH = os.path.join('evaluation', 'metrics_report.md')


def _append_report(title, lines):
    """追加一节到评测报告"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'a', encoding='utf-8') as f:
        f.write(f"\n## {title}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"\n[报告已写入] {REPORT_PATH}")


def _pct(values, p):
    """百分位数"""
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(len(values) * p / 100)))
    return values[idx]


def _fmt_stats(values, unit='ms', nd=1):
    if not values:
        return f"无样本"
    return (f"均值 {statistics.mean(values):.{nd}f}{unit} | "
            f"中位 {statistics.median(values):.{nd}f}{unit} | "
            f"P95 {_pct(values, 95):.{nd}f}{unit} | "
            f"最大 {max(values):.{nd}f}{unit}")


# ==================== perf：推理性能与延迟 ====================

def cmd_perf(args):
    import cv2
    import numpy as np
    from core.config_manager import ConfigManager
    from core.detection_engine import DetectionEngine
    from core.risk_evaluator import RiskEvaluator
    from core.scene_analyzer import SceneAnalyzer
    from core.tracker import SimpleTracker
    from core.traffic_light import TrafficLightClassifier
    from core.agent import BlindGuardAgent
    from core.labels import get_class_cn

    cfg = ConfigManager('config.yaml')
    model_cfg = cfg.get_section('model')
    risk_cfg = cfg.get_section('risk')
    class_mapping = cfg.get('class_mapping', {}) or {}

    detector = DetectionEngine(
        model_path=model_cfg.get('path', 'best.pt'),
        confidence_threshold=model_cfg.get('confidence_threshold', 0.5),
        iou_threshold=model_cfg.get('iou_threshold', 0.45),
        device=model_cfg.get('device', ''),
        img_size=model_cfg.get('img_size', 640),
    )
    risk = RiskEvaluator(
        risk_thresholds={'distance': risk_cfg.get('thresholds', {})},
        high_risk_classes=risk_cfg.get('high_risk_classes'),
        medium_risk_classes=risk_cfg.get('medium_risk_classes'),
        low_risk_classes=risk_cfg.get('low_risk_classes'),
        class_mapping=class_mapping,
    )
    scene = SceneAnalyzer(class_mapping=class_mapping)
    tracker = SimpleTracker()
    light_clf = TrafficLightClassifier()

    # 模板播报路径计时（排除 LLM 网络波动；LLM 通常增加 1-3s，另行说明）
    agent_cfg = cfg.get_section('agent')
    agent_cfg.setdefault('llm', {})['api_key'] = ''
    agent = BlindGuardAgent(agent_cfg, class_mapping=class_mapping)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {args.source}")
        sys.exit(1)

    # 资源占用采样
    cpu_samples, mem_samples, psutil = [], [], None
    try:
        import psutil
        psutil = psutil.Process()
    except ImportError:
        print("[提示] 未安装 psutil，跳过 CPU/内存采样（pip install psutil）")

    t_detect, t_light, t_track, t_risk, t_scene, t_encode = [], [], [], [], [], []
    t_announce = []
    frame_count = 0
    total_target = args.frames
    t_start = time.perf_counter()

    print(f"开始性能评测: {args.source}, 目标帧数 {total_target} ...")
    while frame_count < total_target:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()

        for d in detections:
            if d.get('class_name') in ('traffic_light', 'traffic light'):
                d['light_state'] = light_clf.classify(frame, d.get('bbox'))['state']
        t2 = time.perf_counter()

        tracker.update(detections)
        t3 = time.perf_counter()

        fh, fw = frame.shape[:2]
        risk_results = risk.evaluate(detections, frame_width=fw)
        overall = risk.get_overall_risk(risk_results)
        t4 = time.perf_counter()

        scene.analyze(frame, detections)
        t5 = time.perf_counter()

        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        t6 = time.perf_counter()

        enriched = []
        for r in risk_results:
            det = dict(r)
            det['class_name_cn'] = get_class_cn(det['class_name'], class_mapping)
            enriched.append(det)

        # 播报延迟采样（最多 8 次，避免拉长评测时间）
        announce_dets = agent.should_announce(enriched)
        if announce_dets and len(t_announce) < 8:
            agent.update_memory(enriched, overall['overall_level'], fw, fh)
            ta = time.perf_counter()
            agent.generate_announcement(announce_dets, overall['overall_level'], fw, fh)
            t_announce.append((time.perf_counter() - ta) * 1000)

        if psutil:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            mem_samples.append(psutil.memory_info().rss / (1024 * 1024))

        t_detect.append((t1 - t0) * 1000)
        t_light.append((t2 - t1) * 1000)
        t_track.append((t3 - t2) * 1000)
        t_risk.append((t4 - t3) * 1000)
        t_scene.append((t5 - t4) * 1000)
        t_encode.append((t6 - t5) * 1000)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count}/{total_target} 帧")

    total_s = time.perf_counter() - t_start
    cap.release()

    full_pipeline = [a + b + c + d + e + g for a, b, c, d, e, g in
                     zip(t_detect, t_light, t_track, t_risk, t_scene, t_encode)]
    fps_measured = frame_count / total_s if total_s > 0 else 0

    lines = [
        f"- 视频源: `{args.source}`，评测帧数: {frame_count}",
        f"- **实测处理帧率: {fps_measured:.1f} FPS**（不含人为节流）",
        f"- 单帧完整管线耗时: {_fmt_stats(full_pipeline)}",
        f"- 目标检测: {_fmt_stats(t_detect)}",
        f"- 红绿灯状态识别: {_fmt_stats(t_light, nd=2)}",
        f"- 目标跟踪: {_fmt_stats(t_track, nd=2)}",
        f"- 风险评估: {_fmt_stats(t_risk, nd=2)}",
        f"- 场景分析: {_fmt_stats(t_scene, nd=2)}",
        f"- JPEG 编码: {_fmt_stats(t_encode, nd=2)}",
        f"- 模板播报生成延迟（无 LLM）: {_fmt_stats(t_announce)}",
        "- 注: 在线 LLM 播报在模板路径之上增加网络往返（实测 DeepSeek 约 1-3 秒，"
        "可配置 agent.llm.timeout 上限 8 秒），由异步线程承担，不阻塞视频流。",
    ]
    if cpu_samples:
        lines.append(f"- CPU 占用: 均值 {statistics.mean(cpu_samples):.1f}%，"
                     f"峰值 {max(cpu_samples):.1f}%")
        lines.append(f"- 内存占用: 均值 {statistics.mean(mem_samples):.0f} MB，"
                     f"峰值 {max(mem_samples):.0f} MB")
    else:
        lines.append("- CPU/内存: 未安装 psutil，未采样")

    _append_report(f"推理性能与延迟（perf）", lines)
    for line in lines:
        print("  " + line)


# ==================== accuracy：mAP 精度 ====================

def cmd_accuracy(args):
    if not os.path.exists(args.data):
        print(f"[错误] 数据集配置不存在: {args.data}")
        print("请先准备 YOLO 格式测试集（可运行: python evaluate.py bootstrap 抽帧生成伪标签，")
        print("人工校正后编写 data yaml），格式示例：")
        print("""
path: evaluation/dataset   # 数据集根目录
val: images/val            # 校正后的测试图目录
names:
  0: car
  1: bicycle
  2: manhole_cover
  3: public_facility
  4: tactile_paving
  5: traffic_light""")
        sys.exit(1)

    try:
        from ultralytics import YOLOv10 as YOLOModel
    except ImportError:
        from ultralytics import YOLO as YOLOModel
    from core.config_manager import ConfigManager

    model_path = ConfigManager('config.yaml').get('model.path', 'best.pt')
    print(f"加载模型: {model_path}，评测数据: {args.data}")
    model = YOLOModel(model_path)
    metrics = model.val(data=args.data, imgsz=640, verbose=False)

    box = metrics.box
    lines = [
        f"- 模型: `{model_path}`，测试集: `{args.data}`",
        f"- **mAP@0.5: {box.map50 * 100:.1f}**",
        f"- **mAP@0.5:0.95: {box.map * 100:.1f}**",
        "",
        "| 类别 | AP@0.5 | AP@0.5:0.95 |",
        "|---|---|---|",
    ]
    names = metrics.names or {}
    for i, cls_idx in enumerate(getattr(box, 'ap_class_index', []) or []):
        ap50 = box.ap50[i] * 100 if box.ap50 is not None else 0
        ap = box.ap[i] * 100 if box.ap is not None else 0
        lines.append(f"| {names.get(int(cls_idx), cls_idx)} | {ap50:.1f} | {ap:.1f} |")

    _append_report("模型精度（accuracy）", lines)
    for line in lines:
        print("  " + line)


# ==================== light：红绿灯识别准确率 ====================

def cmd_light(args):
    import cv2
    import numpy as np
    from core.traffic_light import TrafficLightClassifier

    clf = TrafficLightClassifier()
    confusion = {}  # 真实 -> {预测: n}

    if args.dir:
        classes = ['red', 'green', 'yellow']
        total, correct = 0, 0
        for true_state in classes:
            sub = os.path.join(args.dir, true_state)
            if not os.path.isdir(sub):
                print(f"[提示] 缺少子目录 {sub}，跳过")
                continue
            confusion.setdefault(true_state, {})
            for name in os.listdir(sub):
                path = os.path.join(sub, name)
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)  # 兼容中文路径
                if img is None:
                    continue
                pred = clf.classify(img, [0, 0, img.shape[1], img.shape[0]])['state']
                confusion[true_state][pred] = confusion[true_state].get(pred, 0) + 1
                total += 1
                if pred == true_state:
                    correct += 1
        if total == 0:
            print("[错误] 未找到测试图片，目录结构应为 crops/red/*.jpg 等")
            sys.exit(1)
        lines = [f"- 真实裁剪图评测，样本数: {total}",
                 f"- **红绿灯状态识别准确率: {correct / total * 100:.1f}%**", ""]
    else:
        n = max(10, args.synthetic)
        rng = np.random.default_rng(42)
        # BGR 颜色（贴近真实灯色亮度，并加噪声增强）
        colors = {'red': (60, 60, 255), 'green': (80, 220, 80), 'yellow': (60, 215, 235)}
        total, correct = 0, 0
        for true_state, bgr in colors.items():
            confusion.setdefault(true_state, {})
            for _ in range(n):
                img = np.full((120, 120, 3), 30, dtype=np.uint8)
                cx, cy = int(rng.integers(30, 90)), int(rng.integers(30, 90))
                r = int(rng.integers(12, 24))
                cv2.circle(img, (cx, cy), r, bgr, -1)
                noise = rng.integers(0, 25, img.shape, dtype=np.uint8)
                img = cv2.add(img, noise)
                pred = clf.classify(img, [0, 0, 120, 120])['state']
                confusion[true_state][pred] = confusion[true_state].get(pred, 0) + 1
                total += 1
                if pred == true_state:
                    correct += 1
        lines = [f"- 合成图像自检（每类 {n} 张，含噪声），样本数: {total}",
                 f"- **红绿灯状态识别准确率: {correct / total * 100:.1f}%**", ""]

    lines.append("| 真实\\预测 | red | yellow | green | unknown |")
    lines.append("|---|---|---|---|---|")
    for true_state in ['red', 'yellow', 'green']:
        row = confusion.get(true_state, {})
        lines.append(f"| {true_state} | {row.get('red', 0)} | {row.get('yellow', 0)} "
                     f"| {row.get('green', 0)} | {row.get('unknown', 0)} |")
    lines.append("")
    lines.append("- 注: 合成自检仅验证分类器逻辑；真实场景准确率请用 --dir 提供裁剪图评测，"
                 "逆光/遮挡样本的 unknown 率即为系统保守沉默率。")

    _append_report("红绿灯状态识别（light）", lines)
    for line in lines:
        print("  " + line)


# ==================== bootstrap：抽帧 + 伪标签 ====================

def cmd_bootstrap(args):
    import cv2
    from core.config_manager import ConfigManager
    from core.detection_engine import DetectionEngine

    model_cfg = ConfigManager('config.yaml').get_section('model')
    detector = DetectionEngine(
        model_path=model_cfg.get('path', 'best.pt'),
        confidence_threshold=model_cfg.get('confidence_threshold', 0.5),
        iou_threshold=model_cfg.get('iou_threshold', 0.45),
        img_size=model_cfg.get('img_size', 640),
    )

    img_dir = os.path.join('evaluation', 'dataset', 'images')
    lbl_dir = os.path.join('evaluation', 'dataset', 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.source}")
        sys.exit(1)

    names = detector.get_class_names()  # id -> name
    idx2id = {i: int(cid) for i, cid in enumerate(sorted(names.keys()))}

    saved, frame_idx = 0, 0
    while saved < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride == 0:
            fh, fw = frame.shape[:2]
            detections = detector.detect(frame)
            lines = []
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cx = ((x1 + x2) / 2) / fw
                cy = ((y1 + y2) / 2) / fh
                w = (x2 - x1) / fw
                h = (y2 - y1) / fh
                cls_id = int(d['class_id'])
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            img_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_name), frame)
            with open(os.path.join(lbl_dir, img_name.replace('.jpg', '.txt')),
                      'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            saved += 1
        frame_idx += 1
    cap.release()

    class_names_block = '\n'.join(f"  {int(k)}: {v}" for k, v in sorted(names.items()))
    print(f"[完成] 已抽取 {saved} 帧到 {img_dir}，伪标签写入 {lbl_dir}")
    print("下一步：")
    print("  1. 用 labelImg / CVAT 打开上面两个目录，校正伪标签（重点核对井盖/盲道/红绿灯）")
    print("  2. 划分 images/val，编写 data yaml（参考 evaluate.py accuracy 的帮助输出）")
    print("  3. 运行: python evaluate.py accuracy --data data_eval.yaml")
    print(f"模型类别表（供 data yaml 使用）：\n{class_names_block}")


def main():
    parser = argparse.ArgumentParser(description='BlindGuard 答辩指标评测')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('perf', help='推理性能与播报延迟')
    p.add_argument('--source', default='uploads/video.mp4')
    p.add_argument('--frames', type=int, default=300)
    p.set_defaults(func=cmd_perf)

    p = sub.add_parser('accuracy', help='模型精度 mAP（需 YOLO 格式测试集）')
    p.add_argument('--data', required=True, help='data yaml 路径')
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser('light', help='红绿灯识别准确率')
    p.add_argument('--dir', default=None, help='含 red/green/yellow 子目录的裁剪图根目录')
    p.add_argument('--synthetic', type=int, default=60, help='无真实数据时的合成自检张数/类')
    p.set_defaults(func=cmd_light)

    p = sub.add_parser('bootstrap', help='抽帧+伪标签，生成测试集底稿')
    p.add_argument('--source', default='uploads/video.mp4')
    p.add_argument('--stride', type=int, default=30, help='每 N 帧抽 1 帧')
    p.add_argument('--frames', type=int, default=150, help='最多抽取帧数')
    p.set_defaults(func=cmd_bootstrap)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
