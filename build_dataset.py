# -*- coding: utf-8 -*-
"""
BlindGuard v2 数据集组装脚本（7 类）

数据来源与策略（对应 docs/重训练指南.md）：
  1. COCO val2017（真值标注）: person/car/bicycle/traffic light
     -> 类别 6/0/1/5，真实人工标注，行人能力的主力数据
  2. HF 井盖分类数据集（图像级真值）: 每张图必含井盖，
     用旧模型定位出框(conf>=0.5)转为检测框 -> 类别 2
  3. 本地视频伪标注: 仅取定制类 manhole_cover/public_facility/tactile_paving
     (2/3/4) 且 conf>=0.65 的框 -> 保留旧模型定制类能力（注明：质量有限，
     生产级精度需按指南自采标注）

用法:
  python build_dataset.py                # 全流程
  python build_dataset.py --skip-manhole # 跳过井盖定位（无 downloads/manhole 时）
  python build_dataset.py --skip-video   # 跳过视频伪标注

输出: dataset/blindguard_v2/{images,labels}/{train,val} + data_blindguard_v2.yaml
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile

CLASS_NAMES = {
    0: 'car', 1: 'bicycle', 2: 'manhole_cover', 3: 'public_facility',
    4: 'tactile_paving', 5: 'traffic_light', 6: 'person',
}
# COCO 类别 id -> 我们的类别 id
COCO_MAP = {1: 6, 3: 0, 2: 1, 10: 5}  # person, car, bicycle, traffic light

ROOT = 'dataset/blindguard_v2'
VAL_RATIO = 0.15
random.seed(42)


def _ensure_dirs():
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)


def _count_classes():
    counts = {v: 0 for v in CLASS_NAMES.values()}
    for split in ('train', 'val'):
        lbl_dir = os.path.join(ROOT, 'labels', split)
        if not os.path.isdir(lbl_dir):
            continue
        for fn in os.listdir(lbl_dir):
            for line in open(os.path.join(lbl_dir, fn), encoding='utf-8'):
                if line.strip():
                    cid = int(line.split()[0])
                    name = CLASS_NAMES.get(cid)
                    if name:
                        counts[name] += 1
    return counts


def _add_sample(img_src, lines, split):
    """读取图片统一转 jpg 写入（文件名消毒），失败返回 None"""
    import cv2
    import numpy as np
    base = os.path.splitext(os.path.basename(img_src))[0]
    safe = ''.join(ch for ch in base if ch.isalnum())[:60]
    if not safe:
        return None
    name = f"{split}_{safe}"
    img = cv2.imdecode(np.fromfile(img_src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    cv2.imwrite(os.path.join(ROOT, 'images', split, name + '.jpg'), img)
    with open(os.path.join(ROOT, 'labels', split, name + '.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return name


def _split_pick():
    return 'val' if random.random() < VAL_RATIO else 'train'


def step_coco():
    """COCO val2017 -> YOLO（person/car/bicycle/traffic light）"""
    ann_path = 'downloads/coco/annotations/instances_val2017.json'
    img_dir = 'downloads/coco/val2017'
    if not os.path.exists(ann_path):
        print('[跳过] COCO 标注不存在')
        return
    print('[1/3] 转换 COCO val2017 ...')
    coco = json.load(open(ann_path, encoding='utf-8'))
    cat2ours = {c['id']: COCO_MAP[c['id']] for c in coco['categories']
                if c['id'] in COCO_MAP}
    imgs = {im['id']: im for im in coco['images']}
    by_image = {}
    for ann in coco['annotations']:
        cid = ann['category_id']
        if cid in cat2ours and not ann.get('iscrowd'):
            by_image.setdefault(ann['image_id'], []).append(ann)

    n = 0
    for image_id, anns in by_image.items():
        info = imgs[image_id]
        src = os.path.join(img_dir, info['file_name'])
        if not os.path.exists(src):
            continue
        w, h = info['width'], info['height']
        lines = []
        for a in anns:
            x, y, bw, bh = a['bbox']
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cat2ours[a['category_id']]} {cx:.6f} {cy:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        if lines:
            _add_sample(src, lines, _split_pick())
            n += 1
    print(f"    COCO 样本: {n} 张")


def step_manhole(model):
    """井盖分类图 + 旧模型定位 -> 检测框（类别 2）"""
    img_dir = 'downloads/manhole/train_x/train/manhole'
    if not os.path.isdir(img_dir):
        print('[跳过] 井盖数据未下载')
        return
    print('[2/3] 井盖分类图定位转检测（图像级真值 + 模型定位）...')
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:1500]
    kept, n = 0, 0
    for src in files:
        results = model.predict(source=src, conf=0.5, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        # 取置信度最高的 manhole_cover 框作为定位
        best_i, best_conf = None, -1
        for i in range(len(r.boxes)):
            cid = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            if model.names.get(cid) == 'manhole_cover' and conf > best_conf:
                best_i, best_conf = i, conf
        if best_i is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[best_i]]
        ih, iw = r.orig_shape
        cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
        lines = [f"2 {cx:.6f} {cy:.6f} {(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}"]
        _add_sample(src, lines, _split_pick())
        kept += 1
        n += 1
        if kept >= 1200:
            break
    print(f"    井盖定位样本: {kept}/{n} 张（模型定位成功率）")


def step_video(model):
    """本地视频伪标注：仅定制类 2/3/4，conf>=0.65"""
    import cv2
    video = 'uploads/video.mp4'
    if not os.path.exists(video):
        print('[跳过] 视频不存在')
        return
    print('[3/3] 视频伪标注（仅井盖/公共设施/盲道，conf>=0.65）...')
    cap = cv2.VideoCapture(video)
    keep_names = {'manhole_cover': 2, 'public_facility': 3, 'tactile_paving': 4}
    frame_idx, saved = 0, 0
    tmp_dir = os.path.join('downloads', 'pseudo_frames')
    os.makedirs(tmp_dir, exist_ok=True)
    while saved < 400:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 45 == 0:
            results = model.predict(source=frame, conf=0.65, verbose=False)
            r = results[0]
            lines = []
            if r.boxes is not None:
                fh, fw = r.orig_shape
                for i in range(len(r.boxes)):
                    cid = int(r.boxes.cls[i])
                    name = model.names.get(cid)
                    if name in keep_names and float(r.boxes.conf[i]) >= 0.65:
                        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i]]
                        lines.append(f"{keep_names[name]} {(x1+x2)/2/fw:.6f} "
                                     f"{(y1+y2)/2/fh:.6f} {(x2-x1)/fw:.6f} "
                                     f"{(y2-y1)/fh:.6f}")
            if lines:
                tmp = os.path.join(tmp_dir, f"f{frame_idx:06d}.jpg")
                cv2.imwrite(tmp, frame)
                _add_sample(tmp, lines, _split_pick())
                os.remove(tmp)
                saved += 1
        frame_idx += 1
    cap.release()
    print(f"    视频伪标注样本: {saved} 张")


def finalize():
    counts = _count_classes()
    total = sum(1 for s in ('train', 'val')
                for _ in os.listdir(os.path.join(ROOT, 'images', s)))
    yaml_path = 'data_blindguard_v2.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(ROOT)}\ntrain: images/train\n"
                f"val: images/val\n\nnames:\n")
        for cid, name in CLASS_NAMES.items():
            f.write(f"  {cid}: {name}\n")
    print(f"\n数据集组装完成: {total} 张 -> {ROOT}")
    print("各类别实例数:")
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = ' [!]' if c < 100 else ''
        print(f"  {name:16s} {c}{flag}")
    print(f"配置已写入: {yaml_path}")
    print("下一步: python train.py --data data_blindguard_v2.yaml --device 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-manhole', action='store_true')
    parser.add_argument('--skip-video', action='store_true')
    args = parser.parse_args()

    _ensure_dirs()

    # 解压 COCO
    coco_zip = 'downloads/coco/val2017.zip'
    ann_zip = 'downloads/coco/annotations.zip'
    if os.path.exists(coco_zip) and not os.path.isdir('downloads/coco/val2017'):
        print('解压 COCO 图片...')
        with zipfile.ZipFile(coco_zip) as z:
            z.extractall('downloads/coco')
    if os.path.exists(ann_zip) and not os.path.exists(
            'downloads/coco/annotations/instances_val2017.json'):
        print('解压 COCO 标注...')
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall('downloads/coco')

    step_coco()

    need_model = not (args.skip_manhole and args.skip_video)
    if need_model:
        from core.config_manager import ConfigManager
        from core.detection_engine import DetectionEngine
        mc = ConfigManager('config.yaml').get_section('model')
        model = DetectionEngine(
            model_path=mc.get('path', 'best.pt'),
            confidence_threshold=0.5,
            img_size=mc.get('img_size', 640),
        ).model  # 直接用 ultralytics 原生接口拿 boxes
        if not args.skip_manhole:
            step_manhole(model)
        if not args.skip_video:
            step_video(model)

    finalize()


if __name__ == '__main__':
    main()
