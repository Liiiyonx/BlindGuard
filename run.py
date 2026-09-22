# -*- coding: utf-8 -*-
"""BlindGuard 智能导盲系统启动脚本。

使用方法：
    python run.py

功能：
1. 检查依赖是否安装
2. 按 config.yaml 校验主模型与辅助模型
3. 使用配置中的地址和端口启动 Web 服务
"""

import sys

from core.config_manager import ConfigManager
from core.model_validation import model_paths_for_report, missing_model_files
from core.version import VERSION


def check_dependencies():
    """检查依赖是否安装"""
    print("=" * 60)
    print("BlindGuard 智能导盲系统 - 环境检查")
    print("=" * 60)

    required_packages = {
        'flask': 'Flask',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics',
        'pyttsx3': 'pyttsx3',
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

    if missing:
        print(f"\n[警告] 缺少 {len(missing)} 个必需依赖: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print(f"\n[OK] 所有必需依赖已安装 ({len(installed)}个)")
    return True


def check_models(config):
    """按配置检查实际使用的主模型与辅助模型。"""
    print("\n" + "=" * 60)
    print("模型文件检查")
    print("=" * 60)

    entries = list(model_paths_for_report(config))
    if not entries:
        print("  [!!] 未配置任何模型路径")
        return False

    missing_paths = {path for _, path in missing_model_files(config)}
    for role, model_path in entries:
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {role}: {model_path} ({size_mb:.1f}MB)")
        else:
            print(f"  [!!] {role}: {model_path} 不存在")

    if not missing_paths:
        print("\n[OK] 所有模型文件就绪")
        return True
    print("\n[错误] 配置引用的模型文件缺失，系统无法启动")
    return False


def check_config():
    """加载并检查配置文件。"""
    print("\n" + "=" * 60)
    print("配置文件检查")
    print("=" * 60)

    try:
        config = ConfigManager()
    except Exception as exc:
        print(f"  [!!] 配置加载失败: {exc}")
        return None

    if not config.config_path.exists():
        print(f"  [!!] 配置文件不存在: {config.config_path}")
        return None

    print(f"  [OK] 配置文件: {config.config_path.resolve()}")
    if not config.validate():
        print("  [!!] 配置校验失败")
        return None
    print("  [OK] 配置校验通过")
    return config


def start_server(config):
    """启动服务器"""
    print("\n" + "=" * 60)
    print("启动 BlindGuard 服务器")
    print("=" * 60)

    try:
        from app import BlindGuardApp

        print("\n正在初始化系统...")
        app = BlindGuardApp(str(config.config_path))

        print("\n" + "-" * 60)
        print("[OK] 系统初始化完成！")
        print("-" * 60)
        display_host = "localhost" if app.server_host in ("0.0.0.0", "::") else app.server_host
        print(f"\n访问地址: http://{display_host}:{app.server_port}")
        if app.server_token:
            print("远程访问令牌已启用")
        print("按 Ctrl+C 停止服务器\n")

        # 启动服务器
        app.run()

    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")
        print("\n请检查:")
        print("1. 依赖是否正确安装")
        print("2. config.yaml 中的模型路径是否存在")
        print(f"3. 端口 {config.get('server.port', 5000)} 是否被占用")
        sys.exit(1)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(f"   BlindGuard 智能导盲系统 V{VERSION}")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖，然后重新运行此脚本")
        sys.exit(1)

    # 检查配置
    config = check_config()
    if config is None:
        print("\n请先修复配置文件，然后重新运行此脚本")
        sys.exit(1)

    # 按配置检查模型
    if not check_models(config):
        sys.exit(1)

    # 启动服务器
    start_server(config)


if __name__ == '__main__':
    main()
