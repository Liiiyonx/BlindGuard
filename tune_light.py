# -*- coding: utf-8 -*-
"""
灯色识别器参数调优：在 BDD 真实红绿灯上调 HSV 参数（调参/测试五五分，防过拟合）
用法: python tune_light.py
"""
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
truth = json.load(open('downloads/bdd/light_state_val.json', encoding='utf-8'))

# 1. 缓存所有灯的 HSV 裁剪
samples = []
for entry in truth:
    img = cv2.imread(entry['file'])
    if img is None:
        continue
    h, w = img.shape[:2]
    for lt in entry['lights']:
        x, y, bw, bh = lt['bbox']
        x1, y1, x2, y2 = int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h)
        if x2-x1 < 4 or y2-y1 < 4:
            continue
        crop = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        samples.append(({'R': 'red', 'Y': 'yellow', 'G': 'green'}[lt['color']], hsv))
print(f'缓存样本: {len(samples)}')

import random
random.seed(42)
random.shuffle(samples)
half = len(samples) // 2
tune_set, test_set = samples[:half], samples[half:]


def classify(hsv, ranges, min_pixels):
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    best_state, best_count = 'unknown', 0
    for state, rgs in ranges.items():
        count = 0
        for h_lo, h_hi, s_min, v_min in rgs:
            mask = (hue >= h_lo) & (hue <= h_hi) & (sat >= s_min) & (val >= v_min)
            count += int(mask.sum())
        if count > best_count:
            best_state, best_count = state, count
    if best_count >= min_pixels:
        return best_state
    return 'unknown'


def make_ranges(y_lo, y_hi, r_hi, s_min, v_min):
    return {
        'red': [(0, r_hi, s_min, v_min), (360 - r_hi if False else 180 - r_hi + 10, 180, s_min, v_min)],
        'yellow': [(y_lo, y_hi, s_min, v_min)],
        'green': [(40, 90, max(40, s_min - 20), max(60, v_min - 10))],
    }


def acc(samples_set, ranges, min_pixels):
    correct = 0
    for truth_state, hsv in samples_set:
        if classify(hsv, ranges, min_pixels) == truth_state:
            correct += 1
    return correct / len(samples_set) * 100


DEFAULT = {
    'red': [(0, 10, 80, 90), (170, 180, 80, 90)],
    'yellow': [(15, 35, 80, 90)],
    'green': [(40, 90, 60, 80)],
}
print(f"默认参数: tune {acc(tune_set, DEFAULT, 20):.1f}%  test {acc(test_set, DEFAULT, 20):.1f}%")

# 2. 网格搜索（在 tune 集上）
best = (None, -1)
for y_lo, y_hi in [(15, 35), (20, 40), (12, 30), (20, 45)]:
    for r_hi in (8, 10, 13):
        for s_min in (60, 80, 100):
            for v_min in (70, 90, 110):
                for mp in (10, 20, 35):
                    ranges = make_ranges(y_lo, y_hi, r_hi, s_min, v_min)
                    a = acc(tune_set, ranges, mp)
                    if a > best[1]:
                        best = ((y_lo, y_hi, r_hi, s_min, v_min, mp), a)
print(f"最优参数: {best[0]}  tune 准确率 {best[1]:.1f}%")

y_lo, y_hi, r_hi, s_min, v_min, mp = best[0]
best_ranges = make_ranges(y_lo, y_hi, r_hi, s_min, v_min)
print(f" held-out test: {acc(test_set, best_ranges, mp):.1f}%  (默认参数同测试集 {acc(test_set, DEFAULT, 20):.1f}%)")
print(f"min_pixels={mp}")
print(f"yellow: {(y_lo, y_hi, s_min, v_min)}  red: [(0,{r_hi},{s_min},{v_min}),({180-r_hi+10},180,{s_min},{v_min})]")
print(f"green: (40, 90, {max(40, s_min-20)}, {max(60, v_min-10)})")
