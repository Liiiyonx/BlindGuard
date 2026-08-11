# -*- coding: utf-8 -*-
"""
BlindGuard 智能导盲系统 V1.0 - 启动脚本

使用方法：
    python run.py

功能：
1. 检查依赖是否安装
2. 检查模型文件是否存在
3. 启动Web服务
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """检查依赖是否安装"""
    print("=" * 60)
    print("BlindGuard 智能导盲系统 - 环境检查")
    print("=" * 60)

    required_packages = {
        'flask': 'Flask',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'ultralytics': 'Ultralytics',
        'pyttsx3': 'pyttsx3'
    }

    optional_packages = {
        'yaml': 'PyYAML'
    }

    missing = []
    installed = []

    # 检查必需包
    print("\n必需依赖:")
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"  [OK] {name:20s} 已安装")
            installed.append(name)
        except ImportError:
            print(f"  [!!] {name:20s} 未安装")
            missing.append(name)

    # 检查可选包
    print("\n可选依赖:")
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"  [OK] {name:20s} 已安装")
        except ImportError:
            print(f"  [--] {name:20s} 未安装（可选）")

    if missing:
        print(f"\n[警告] 缺少 {len(missing)} 个必需依赖: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print(f"\n[OK] 所有必需依赖已安装 ({len(installed)}个)")
    return True


def check_models():
    """检查模型文件"""
    print("\n" + "=" * 60)
    print("模型文件检查")
    print("=" * 60)

    models = {
        'best.pt': '自定义训练模型',
        'yolov10n.pt': 'YOLOv10基础模型'
    }

    all_exist = True
    for model_file, description in models.items():
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  [OK] {model_file:20s} ({size_mb:.1f}MB) - {description}")
        else:
            print(f"  [!!] {model_file:20s} 不存在 - {description}")
            all_exist = False

    if all_exist:
        print("\n[OK] 所有模型文件就绪")
    else:
        print("\n[警告] 部分模型文件缺失，系统可能无法正常运行")

    return all_exist


def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("配置文件检查")
    print("=" * 60)

    config_files = ['config.yaml', 'config.yml', 'config.json']

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  [OK] {config_file} 已找到")
            return True

    print("  [!!] 未找到配置文件")
    print("  将使用默认配置")
    return False


def start_server():
    """启动服务器"""
    print("\n" + "=" * 60)
    print("启动 BlindGuard 服务器")
    print("=" * 60)

    try:
        from app import BlindGuardApp

        print("\n正在初始化系统...")
        app = BlindGuardApp()

        print("\n" + "-" * 60)
        print("[OK] 系统初始化完成！")
        print("-" * 60)
        print("\n访问地址: http://localhost:5000")
        print("按 Ctrl+C 停止服务器\n")

        # 启动服务器
        app.run(host='0.0.0.0', port=5000, debug=False)

    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")
        print("\n请检查:")
        print("1. 依赖是否正确安装")
        print("2. 模型文件是否存在")
        print("3. 端口5000是否被占用")
        sys.exit(1)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("   BlindGuard 智能导盲系统 V1.0")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖，然后重新运行此脚本")
        sys.exit(1)

    # 检查模型
    check_models()

    # 检查配置
    check_config()

    # 启动服务器
    start_server()


if __name__ == '__main__':
    main()
