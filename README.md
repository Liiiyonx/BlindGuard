# BlindGuard 智能导盲系统 V1.0

## 系统简介

BlindGuard 是一款基于计算机视觉的智能导盲辅助系统，旨在为视障人士提供实时环境感知和语音导航服务。系统通过摄像头捕获周围环境，利用深度学习算法进行目标检测，并根据风险等级提供语音预警。

## 核心功能

### 1. 实时目标检测
- 基于YOLOv8/v10的实时目标检测
- 支持80+种常见物体识别
- 优化检测参数，平衡速度与精度

### 2. 智能风险评估
- **多维度评估算法**：综合考虑目标类型、距离、位置
- **动态风险等级**：极高/高/中/低/安全五级评估
- **可配置阈值**：支持运行时调整风险参数

### 3. 语音播报系统
- **优先级队列**：高风险信息优先播报
- **智能冷却**：避免重复播报相同内容
- **异步处理**：不阻塞主检测流程

### 4. 场景分析
- **目标分布统计**：按类别和空间位置分析
- **运动趋势检测**：识别新出现/消失的目标
- **场景类型判断**：空旷/简单/复杂/拥挤/危险

### 5. Web可视化界面
- **实时视频流**：显示摄像头画面和检测结果
- **状态监控**：FPS、目标数量、风险等级
- **控制面板**：系统启动/停止、摄像头控制

## 系统架构

```
BlindGuard/
├── app.py                    # 主应用入口
├── config.json              # 系统配置
├── requirements.txt         # 依赖清单
├── core/                    # 核心模块
│   ├── detection_engine.py  # 检测引擎
│   ├── risk_evaluator.py    # 风险评估
│   ├── voice_announcer.py   # 语音播报
│   ├── scene_analyzer.py    # 场景分析
│   └── config_manager.py    # 配置管理
├── templates/               # 前端模板
│   └── index.html           # Web界面
├── logs/                    # 日志目录
└── uploads/                 # 上传目录
```

## 快速开始

### 1. 环境要求
- Python 3.8+
- 摄像头（可选，支持视频文件）
- 操作系统：Windows/Linux/macOS

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型
```bash
# YOLOv8n 模型（约6MB）
# 首次运行时会自动下载
```

### 4. 启动系统
```bash
python app.py
```

### 5. 访问界面
打开浏览器访问：http://localhost:5000

## 配置说明

### 系统配置 (config.json)

```json
{
  "model": {
    "path": "yolov8n.pt",           // 模型路径
    "confidence_threshold": 0.45,    // 置信度阈值
    "iou_threshold": 0.5            // IoU阈值
  },
  "risk": {
    "thresholds": {
      "distance": {
        "very_close": 0.20,         // 非常近阈值
        "close": 0.10,              // 近阈值
        "medium": 0.04,             // 中等阈值
        "far": 0.01                 // 远阈值
      }
    }
  },
  "voice": {
    "rate": 160,                    // 语速
    "volume": 1.0,                  // 音量
    "cooldown": 3.0                 // 冷却时间
  }
}
```

### 环境变量
支持通过环境变量覆盖配置：
```bash
export BLINDGUARD_MODEL_PATH="yolov8s.pt"
export BLINDGUARD_VOICE_RATE=180
```

## API接口

### 系统控制
- `POST /api/system/start` - 启动系统
- `POST /api/system/stop` - 停止系统
- `GET /api/system/status` - 获取系统状态

### 摄像头控制
- `POST /api/camera/start` - 开启摄像头
- `POST /api/camera/stop` - 关闭摄像头

### 检测功能
- `POST /api/detect` - 单帧图像检测
- `GET /api/detections` - 获取当前检测结果
- `GET /video_feed` - 实时视频流

## 风险等级说明

| 等级 | 颜色 | 说明 | 建议动作 |
|------|------|------|----------|
| 极高风险 | 🔴 红色 | 目标非常近且危险 | 立即停下 |
| 高风险 | 🟠 橙红色 | 目标较近需避让 | 注意避让 |
| 中等风险 | 🟡 橙色 | 需要注意观察 | 注意观察 |
| 低风险 | 🔵 蓝色 | 距离较远 | 留意周围 |
| 安全 | 🟢 绿色 | 无明显障碍 | 正常通行 |

## 开发说明

### 核心算法

#### 风险评估算法
```python
# 多维度加权评估
total_score = (
    class_score * 0.35 +      # 目标类别权重
    distance_score * 0.30 +   # 距离权重
    position_score * 0.20 +   # 位置权重
    movement_score * 0.15     # 运动权重
)
```

#### 场景分析算法
- 基于目标数量和类型判断场景复杂度
- 统计目标在画面中的空间分布
- 对比前后帧分析运动趋势

### 扩展开发

#### 添加新的检测类别
```python
# 在 risk_evaluator.py 中添加风险分值
CLASS_RISK_SCORES = {
    'new_class': 0.6,  # 添加新类别
    # ...
}
```

#### 自定义风险阈值
```python
# 修改 config.json 中的阈值配置
{
  "risk": {
    "thresholds": {
      "distance": {
        "very_close": 0.25,  # 调整阈值
        # ...
      }
    }
  }
}
```

## 常见问题

### Q: 摄像头无法打开怎么办？
A: 尝试以下方法：
1. 检查摄像头是否被其他程序占用
2. 修改 `device_id`（0, 1, 2...）
3. 更新摄像头驱动

### Q: 检测速度太慢怎么办？
A: 可以：
1. 使用更小的模型（yolov8n → yolov8s）
2. 降低输入分辨率
3. 使用GPU加速

### Q: 语音播报不工作怎么办？
A: 检查：
1. pyttsx3 是否正确安装
2. 系统是否有可用的语音引擎
3. 音量设置是否为0

## 版本历史

### V1.0 (2024)
- 初始版本发布
- 实现基础目标检测功能
- 实现风险评估算法
- 实现语音播报系统
- 实现Web可视化界面

## 许可证

版权所有 (c) 2024-2025 BlindGuard Team

## 联系方式

- 项目主页：[待添加]
- 问题反馈：[待添加]
- 邮箱：[待添加]

---

**注意**：本系统仅供辅助使用，不能完全替代盲杖或导盲犬等传统辅助工具。在使用本系统时，请始终保持警惕，注意安全。
