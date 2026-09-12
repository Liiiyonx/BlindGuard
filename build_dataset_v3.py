# -*- coding: utf-8 -*-
"""
BlindGuard v3 数据集组装（全新 7 类，与旧模型无关）

类别（全新定义，基于 COCO 全量真值）:
  0: person  1: bicycle  2: car  3: motorcycle
  4: bus     5: truck    6: traffic_light

数据:
  - 训练: COCO train2017 选择性抽取（含 7 类目标的全量非行人图 +
    行人图抽样补齐，总量受磁盘限制 capped）
  - 验证: COCO val2017 全部含 7 类目标的图

用法: python build_dataset_v3.py [--max-total 50000]
输出: dataset/blindguard_v3/ + data_blindguard_v3.yaml
"""

import argparse
import io
import json
import os
import random
import sys

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
               4: 'bus', 5: 'truck', 6: 'traffic_light'}
# COCO category id -> v3 id
COCO_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5, 10: 6}

ROOT = 'dataset/blindguard_v3'
random.seed(42)


def coco_to_yolo_lines(anns, w, h):
    lines = []
    for a in anns:
        x, y, bw, bh = a['bbox']
        lines.append(f"{COCO_MAP[a['category_id']]} {(x+bw/2)/w:.6f} "
                     f"{(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
    return lines


def index_coco(json_path):
    """返回 {image_id: (file_name, w, h, anns)} 仅含 7 类目标"""
    coco = json.load(open(json_path, encoding='utf-8'))
    imgs = {im['id']: im for im in coco['images']}
    by_img = {}
    for a in coco['annotations']:
        if a['category_id'] in COCO_MAP and not a.get('iscrowd'):
            by_img.setdefault(a['image_id'], []).append(a)
    out = {}
    for image_id, anns in by_img.items():
        im = imgs[image_id]
        out[image_id] = (im['file_name'], im['width'], im['height'], anns)
    return out


def write_sample(src_file, zf, lines, split, name):
    import cv2
    import numpy as np
    data = zf.read(src_file)
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False
    cv2.imwrite(os.path.join(ROOT, 'images', split, name + '.jpg'), img)
    with open(os.path.join(ROOT, 'labels', split, name + '.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return True


class MultiFileReader(io.RawIOBase):
    """把 part_0..5.bin 顺序拼接成可 seek 的虚拟文件，避免落盘合并占双倍空间"""

    def __init__(self, paths):
        import bisect
        self._bisect = bisect.bisect_right
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.total = self.offsets[-1]
        self.pos = 0
        self._fh = None
        self._idx = -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.total + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n == -1:
            n = self.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total:
            idx = self._bisect(self.offsets, self.pos) - 1
            if idx != self._idx:
                if self._fh:
                    self._fh.close()
                self._fh = open(self.paths[idx], 'rb')
                self._idx = idx
            self._fh.seek(self.pos - self.offsets[idx])
            avail = self.offsets[idx + 1] - self.pos
            data = self._fh.read(min(n, avail))
            if not data:
                break
            out += data
            self.pos += len(data)
            n -= len(data)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-total', type=int, default=35000)
    args = parser.parse_args()

    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)

    # ---- 验证集: val2017（已解压） ----
    val_json = 'downloads/coco/annotations/instances_val2017.json'
    val_dir = 'downloads/coco/val2017'
    print('[1/2] 组装验证集 (val2017) ...')
    val_index = index_coco(val_json)
    n_val = 0
    for image_id, (fname, w, h, anns) in val_index.items():
        src = os.path.join(val_dir, fname)
        if not os.path.exists(src):
            continue
        lines = coco_to_yolo_lines(anns, w, h)
        name = 'val_' + os.path.splitext(fname)[0]
        cv2_src = src
        import shutil
        shutil.copy(cv2_src, os.path.join(ROOT, 'images', 'val', name + '.jpg'))
        with open(os.path.join(ROOT, 'labels', 'val', name + '.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(lines))
        n_val += 1
    print(f"    验证集: {n_val} 张")

    # ---- 训练集: train2017 分段虚拟读取（part_0..5.bin，不再需要完整 zip） ----
    train_json = 'downloads/coco/annotations/instances_train2017.json'
    if not os.path.exists(train_json):
        print('[错误] train2017 标注缺失，请重新下载 annotations_trainval2017.zip')
        sys.exit(1)

    print('[2/2] 组装训练集 (train2017 分段虚拟读取抽取) ...')
    train_index = index_coco(train_json)
    # 分组: 含非 person 类的图全要；仅 person 的图抽样
    multi, person_only = [], []
    for image_id, (fname, w, h, anns) in train_index.items():
        cats = {a['category_id'] for a in anns}
        if cats - {1}:          # 有 person 以外的目标类
            multi.append(image_id)
        else:
            person_only.append(image_id)
    random.shuffle(person_only)
    budget = max(0, args.max_total - len(multi))
    selected = multi + person_only[:budget]
    print(f"    候选: 多类图 {len(multi)} + 行人图抽样 {min(len(person_only), budget)}"
          f" = {len(selected)} 张")

    import io
    import zipfile
    part_paths = [f'downloads/coco/part_{i}.bin' for i in range(6)]
    reader = io.BufferedReader(MultiFileReader(part_paths), buffer_size=1 << 22)
    zf = zipfile.ZipFile(reader)
    names_in_zip = set(zf.namelist())
    n_train = 0
    for image_id in selected:
        fname, w, h, anns = train_index[image_id]
        zname = 'train2017/' + fname
        if zname not in names_in_zip:
            continue
        lines = coco_to_yolo_lines(anns, w, h)
        name = 'train_' + os.path.splitext(fname)[0]
        if write_sample(zname, zf, lines, 'train', name):
            n_train += 1
        if n_train % 5000 == 0 and n_train:
            print(f"    ...已抽取 {n_train}")
    print(f"    训练集: {n_train} 张")

    # ---- yaml ----
    yaml_path = 'data_blindguard_v3.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(ROOT)}\ntrain: images/train\n"
                f"val: images/val\n\nnames:\n")
        for cid, name in CLASS_NAMES.items():
            f.write(f"  {cid}: {name}\n")

    # ---- 类别统计 ----
    counts = {v: 0 for v in CLASS_NAMES.values()}
    for split in ('train', 'val'):
        for fn in os.listdir(os.path.join(ROOT, 'labels', split)):
            for line in open(os.path.join(ROOT, 'labels', split, fn),
                             encoding='utf-8'):
                if line.strip():
                    counts[CLASS_NAMES[int(line.split()[0])]] += 1
    print(f"\n完成: train {n_train} + val {n_val} -> {ROOT}")
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:14s} {c}")
    print(f"配置: {yaml_path}")
    print("训练: python train.py --data data_blindguard_v3.yaml --weights yolov10n.pt "
          "--epochs 20 --imgsz 640 --device 0 --workers 4 --batch 32 --name blindguard_v3")


if __name__ == '__main__':
    main()
