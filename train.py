# -*- coding: utf-8 -*-
"""
BlindGuard 模型重训练脚本

用法:
  python train.py --data data_blindguard_v2.yaml
      从 COCO 预训练权重训练新模型（推荐：类别有增减时必须走这条）

  python train.py --data data_blindguard_v2.yaml --weights best.pt --epochs 60
      从现有 best.pt 继续微调（仅当类别与 best.pt 完全一致时可用）

  python train.py --data ... --imgsz 640 --device 0 --batch 16
      显存不足时降低 imgsz/指定参数；无 GPU 用 --device cpu（慢）

训练完成后自动在验证集上评测，并打印与当前线上模型的对比和上线步骤。
"""

import argparse
import os
import shutil
import sys

try:
    from ultralytics import YOLOv10 as YOLOModel
except ImportError:
    from ultralytics import YOLO as YOLOModel


def main():
    parser = argparse.ArgumentParser(description='BlindGuard 模型重训练')
    parser.add_argument('--data', required=True, help='数据集 yaml（含 path/train/val/names）')
    parser.add_argument('--weights', default='yolov10n.pt',
                        help='初始权重：新增类别用 yolov10n.pt(COCO预训练)；'
                             '类别不变时可用 best.pt 微调')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=960,
                        help='井盖/盲道为小目标，建议 960；数据量大或显存不足可 640')
    parser.add_argument('--batch', type=int, default=-1, help='-1 为自动批大小')
    parser.add_argument('--device', default='', help="'0' GPU / 'cpu' / '' 自动")
    parser.add_argument('--name', default='blindguard_v2', help='本次训练的运行名')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心轮数')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader 工作进程数；Windows 下 8 易崩，建议 4 或 2')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 数据集配置不存在: {args.data}")
        print("请先按 docs/重训练指南.md 完成数据采集与标注，yaml 模板见指南。")
        sys.exit(1)

    print("=" * 60)
    print(f"开始训练: weights={args.weights} data={args.data}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device or 'auto'}")
    print("=" * 60)

    model = YOLOModel(args.weights)
    import cv2
    cv2.setNumThreads(0)  # 避免 Windows 下 DataLoader 工作进程内 OpenCV 线程冲突崩溃
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
        workers=args.workers,
        plots=True,       # 训练曲线/混淆矩阵输出到 runs/detect/<name>/
    )

    # 训练完成后在验证集上评测新模型
    best_path = os.path.join('runs', 'detect', args.name, 'weights', 'best.pt')
    if not os.path.exists(best_path):
        best_path = os.path.join('runs', 'detect', args.name + str(2), 'weights', 'best.pt')

    print("\n" + "=" * 60)
    print("训练完成，正在验证集上评测新模型 ...")
    new_model = YOLOModel(best_path)
    metrics = new_model.val(data=args.data, imgsz=args.imgsz, verbose=False)
    box = metrics.box
    print(f"  新模型 mAP@0.5 = {box.map50 * 100:.1f}")
    print(f"  新模型 mAP@0.5:0.95 = {box.map * 100:.1f}")
    names = metrics.names or {}
    ap_idx = getattr(box, 'ap_class_index', None)
    ap_idx = list(ap_idx) if ap_idx is not None else []
    for i, cls_idx in enumerate(ap_idx):
        print(f"    {names.get(int(cls_idx), cls_idx):16s} AP50 = {box.ap50[i] * 100:.1f}")

    print("\n上线步骤:")
    print(f"  1. 备份现有模型:  copy best.pt best_backup.pt")
    print(f"  2. 替换模型文件:  copy {best_path} best.pt")
    print(f"  3. 冒烟验证:      python smoke_test.py")
    print(f"  4. 实拍复检:      python evaluate.py collect 后人工抽检对比")
    print(f"  5. 精度报告:      python evaluate.py accuracy --data {args.data}")
    print("=" * 60)


if __name__ == '__main__':
    main()
