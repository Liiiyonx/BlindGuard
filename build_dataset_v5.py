# -*- coding: utf-8 -*-
"""
BlindGuard v5 数据集组装：COCO 35k（v3）+ BDD100K 10k 路域混合

BDD100K（行车视角街拍，贴近部署域）标注映射:
  pedestrian->person(0)  bicycle->bicycle(1)  car->car(2)  motorcycle->motorcycle(3)
  bus->bus(4)  truck->truck(5)  traffic light->traffic_light(6)
  跳过: rider/traffic sign/other vehicle/train/trailer/other person
红绿灯的颜色属性(R/Y/G)单独导出，用于真实验证灯色识别器。

用法: python build_dataset_v5.py
输出: dataset/blindguard_v5/ + data_blindguard_v5.yaml
      dataset/bdd_val/ + data_bdd_val.yaml（BDD 独立验证集）
      downloads/bdd/light_state_val.json（灯色真值）
"""

import json
import os
import random
import shutil
import sys

random.seed(42)

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
               4: 'bus', 5: 'truck', 6: 'traffic_light'}
BDD_MAP = {'pedestrian': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
           'bus': 4, 'truck': 5, 'traffic light': 6}

ROOT = 'dataset/blindguard_v5'
SRC_V3 = 'dataset/blindguard_v3'
BDD_IMG_DIR = 'downloads/bdd'


def main():
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)

    # ---- 1. 复制 v3（COCO 35k）为基底 ----
    print('[1/3] 复制 v3 (COCO 35k) ...')
    if not os.path.isdir(SRC_V3):
        print(f'[错误] {SRC_V3} 不存在（v3 数据集已被清理，需重新运行 build_dataset_v3.py）')
        sys.exit(1)
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            src = os.path.join(SRC_V3, kind, split)
            dst = os.path.join(ROOT, kind, split)
            for fn in os.listdir(src):
                shutil.copy(os.path.join(src, fn), os.path.join(dst, 'coco_' + fn))
    n_coco_train = len(os.listdir(os.path.join(ROOT, 'images', 'train')))
    n_coco_val = len(os.listdir(os.path.join(ROOT, 'images', 'val')))
    print(f'    COCO: train {n_coco_train} + val {n_coco_val}')

    # ---- 2. BDD 转换并入 ----
    print('[2/3] BDD100K 转换（路域真值）...')
    samples = json.load(open('downloads/bdd/samples.json', encoding='utf-8'))['samples']
    light_truth = []
    n_bdd_train = n_bdd_val = 0
    import cv2
    import numpy as np
    for s in samples:
        det_field = s.get('detections')
        if not (isinstance(det_field, dict) and det_field.get('detections')):
            continue
        fname = os.path.basename(s['filepath'])
        src_img = os.path.join(BDD_IMG_DIR, fname)
        if not os.path.exists(src_img):
            continue
        img = cv2.imread(src_img)
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        lights = []
        for det in det_field['detections']:
            label = det.get('label')
            bb = det.get('bounding_box')
            if label not in BDD_MAP or not bb or len(bb) < 4:
                continue
            x, y, bw, bh = bb
            lines.append(f"{BDD_MAP[label]} {x+bw/2:.6f} {y+bh/2:.6f} {bw:.6f} {bh:.6f}")
            if label == 'traffic light' and det.get('trafficLightColor') in 'RYG':
                lights.append({'color': det['trafficLightColor'],
                               'bbox': [x, y, bw, bh]})
        if not lines:
            continue
        split = 'val' if random.random() < 0.10 else 'train'
        name = f"bdd_{os.path.splitext(fname)[0]}"
        cv2.imwrite(os.path.join(ROOT, 'images', split, name + '.jpg'), img)
        with open(os.path.join(ROOT, 'labels', split, name + '.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if split == 'val' and lights:
            light_truth.append({'file': os.path.join(ROOT, 'images', 'val', name + '.jpg'),
                                'lights': lights})
        if split == 'train':
            n_bdd_train += 1
        else:
            n_bdd_val += 1
    print(f'    BDD: train {n_bdd_train} + val {n_bdd_val}')

    # 灯色真值（独立于训练）
    with open('downloads/bdd/light_state_val.json', 'w', encoding='utf-8') as f:
        json.dump(light_truth, f, ensure_ascii=False, indent=1)
    n_lights = sum(len(x['lights']) for x in light_truth)
    print(f'    灯色真值: {n_lights} 个红绿灯 (来自 {len(light_truth)} 张验证图)')

    # ---- 3. yaml ----
    with open('data_blindguard_v5.yaml', 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(ROOT)}\ntrain: images/train\n"
                f"val: images/val\n\nnames:\n")
        for cid, name in CLASS_NAMES.items():
            f.write(f"  {cid}: {name}\n")

    # BDD 独立验证集（用于对比 v4/v5 的路域指标）
    bdd_val_root = 'dataset/bdd_val'
    for sub in ('images/val', 'labels/val'):
        os.makedirs(os.path.join(bdd_val_root, sub), exist_ok=True)
    for fn in os.listdir(os.path.join(ROOT, 'images', 'val')):
        if fn.startswith('bdd_'):
            shutil.copy(os.path.join(ROOT, 'images', 'val', fn),
                        os.path.join(bdd_val_root, 'images', 'val', fn))
            shutil.copy(os.path.join(ROOT, 'labels', 'val', fn.replace('.jpg', '.txt')),
                        os.path.join(bdd_val_root, 'labels', 'val', fn.replace('.jpg', '.txt')))
    with open('data_bdd_val.yaml', 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(bdd_val_root)}\ntrain: images/val\n"
                f"val: images/val\n\nnames:\n")
        for cid, name in CLASS_NAMES.items():
            f.write(f"  {cid}: {name}\n")

    print(f"\n完成: v5 = COCO {n_coco_train}+{n_coco_val} + BDD {n_bdd_train}+{n_bdd_val}")
    print(f"配置: data_blindguard_v5.yaml | BDD 独立验证: data_bdd_val.yaml")
    print("训练: python train.py --data data_blindguard_v5.yaml --weights best_v4.pt "
          "--epochs 6 --imgsz 960 --device 0 --workers 4 --batch 12 --name blindguard_v5")


if __name__ == '__main__':
    main()
