# BlindGuard 智能导盲系统 V2.0

## 系统简介

BlindGuard 是一款基于计算机视觉与大语言模型（LLM）的智能导盲辅助系统，为视障人士提供实时环境感知、风险评估、自然语言语音预警和智能对话服务。

V2.0 在 V1.0 的 CV 管线之上叠加了**智能体（Agent）层**：LLM 作为推理核心，结合时序记忆生成自然语言播报，并支持用户主动提问。

## 系统架构

```
摄像头/视频
    │
    ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  检测引擎    │ → │  风险评估器  │ → │  场景分析器  │
│ Detection   │   │ RiskEvaluator│   │SceneAnalyzer│
│ (YOLO/best) │   │ (多维度评分) │   │ (场景理解)  │
└─────────────┘   └─────────────┘   └──────┬──────┘
                                           │
                    ┌──────────────────────┤
                    ▼                      ▼
            ┌─────────────┐        ┌─────────────┐
            │  智能体核心  │  ←──→  │  时序记忆    │
            │  Agent(LLM) │        │ (滑动窗口)   │
            └──────┬──────┘        └─────────────┘
                   ▼
            ┌─────────────┐
            │  语音播报器  │
            │VoiceAnnouncer│
            │ (优先级队列) │
            └─────────────┘
```

```
BlindGuard/
├── app.py                    # 主应用（装配与调度各核心模块）
├── run.py                    # 启动脚本（含环境检查）
├── smoke_test.py             # 冒烟测试脚本
├── config.yaml               # 唯一配置文件（检测/风险/语音/Agent）
├── .env                      # 私密配置（API Key，已 gitignore）
├── requirements.txt          # 依赖清单
├── core/                     # 核心模块
│   ├── detection_engine.py   # 检测引擎（YOLO 封装）
│   ├── risk_evaluator.py     # 风险评估（多维度加权评分）
│   ├── scene_analyzer.py     # 场景分析（分布/空间/运动）
│   ├── voice_announcer.py    # 语音播报（优先级队列）
│   ├── config_manager.py     # 配置管理（YAML + 环境变量）
│   └── agent.py              # 智能体（LLM 推理 + 记忆 + 对话）
├── templates/
│   └── index.html            # Web 界面（监控 + 智能对话）
├── logs/                     # 日志目录
└── uploads/                  # 视频上传目录
```

## 核心功能

### 1. 实时目标检测（DetectionEngine）
- 基于自定义训练的 YOLO 模型（best.pt）
- 导盲场景定制 6 类：汽车、自行车、井盖、公共设施、盲道、红绿灯
- 可配置置信度/IoU 阈值

### 2. 多维度风险评估（RiskEvaluator）
- **类别 × 距离 × 位置** 三维加权评分
- 五级风险：极高 / 高 / 中 / 低 / 安全
- 风险类别与阈值全部可在 config.yaml 配置

### 3. 场景分析（SceneAnalyzer）
- 目标分布统计、空间方位分析、前后帧运动趋势
- 场景类型判断：空旷/简单/复杂/拥挤/危险

### 4. 智能体层（BlindGuardAgent）⭐ V2.0 新增
- **LLM 自然播报**：不再是固定模板，LLM 结合检测结果、风险等级、方位、时序记忆生成口语化提示（如"注意左边来车，请停下等待"）
- **智能对话**：用户可随时询问"前面有什么""能走吗"，Agent 基于实时检测结果回答
- **时序记忆**：滑动窗口保留最近若干帧的目标变化，支持时序推理
- **离线兜底**：未配置 API Key 或调用失败时自动回退到模板播报与本地对话，系统完整可用
- 支持任意 OpenAI 兼容端点（DeepSeek / 通义千问 / 智谱 GLM / OpenAI）

### 5. 语音播报（VoiceAnnouncer）
- 优先级队列：极高风险插队播报
- 冷却去重 + 异步处理，不阻塞检测
- LLM 播报异步生成，不阻塞视频流

### 6. Web 可视化界面
- 实时视频流、FPS、检测目标数
- 整体风险等级（后端综合评估）、场景类型
- 智能对话面板（含 LLM 在线状态徽章）

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 LLM（可选，推荐）
在项目根目录创建 `.env` 文件（已被 .gitignore 排除，不会提交）：
```bash
BLINDGUARD_AGENT_LLM_API_KEY=你的DeepSeek_API_Key
```
不配置也能运行，系统自动回退到模板播报 + 本地对话。

### 3. 启动系统
```bash
python run.py        # 推荐：带环境检查
# 或
python app.py
```

### 4. 访问界面
打开浏览器访问：http://localhost:5000

### 5. 自检
```bash
python smoke_test.py   # 冒烟测试：模块导入/Agent/接口/模型全链路
```

## 配置说明（config.yaml）

```yaml
detector:            # 检测引擎
  model_path: "best.pt"
  confidence: 0.5
  iou_threshold: 0.45

risk_engine:         # 风险评估
  area_thresholds:   # 距离阈值（面积占比）
    close: 0.15
  high_risk_classes: # 高危类别（含 manhole_cover 等自定义类）
    - "car"
    - "manhole_cover"

agent:               # 智能体
  enabled: true
  llm:
    base_url: "https://api.deepseek.com/v1"
    api_key: ""      # 留空！密钥放 .env
    model: "deepseek-chat"
  announce:
    cooldown: 3.0    # 播报冷却
    memory_size: 8   # 时序记忆窗口
```

所有配置均实际生效（检测阈值、风险类别、语音参数、Agent 行为）。

## API 接口

### 系统控制
- `POST /api/system/start` - 启动系统
- `POST /api/system/stop` - 停止系统
- `GET  /api/system/status` - 系统状态（含整体风险、场景类型、最新播报）

### 摄像头 / 视频
- `POST /api/camera/start` / `POST /api/camera/stop`
- `POST /api/video/upload` - 上传视频检测
- `POST /api/video/pause` / `resume` / `seek` / `speed` / `stop`
- `GET  /video_feed` - 实时视频流

### 检测与智能体
- `GET  /api/detections` - 当前检测结果（含风险等级、风险分值）
- `POST /api/chat` - 智能对话 `{"message": "前面有什么"}`
- `GET  /api/agent/status` - Agent 状态（LLM 是否在线、记忆帧数）

## 风险等级说明

| 等级 | 颜色 | 说明 | 语音优先级 |
|------|------|------|-----------|
| 极高风险 | 🔴 红色闪烁 | 高危目标非常近 | 插队播报 |
| 高风险 | 🟠 橙红 | 需立即避让 | 高 |
| 中等风险 | 🟡 橙色 | 注意观察 | 中 |
| 低风险 | 🔵 蓝色 | 留意周围 | 低 |
| 安全 | 🟢 绿色 | 可正常通行 | - |

## 版本历史

### V2.0 (2026)
- 新增智能体层：LLM 自然播报、智能对话、时序记忆
- app.py 接入全部 core 模块，架构完整闭环
- 配置全部生效（config.yaml 唯一配置源）
- 前端新增对话面板、场景显示、极高风险等级
- 修复 run.py 启动 bug、场景分析运动检测 bug

### V1.0 (2024)
- 基础目标检测、风险评估、语音播报、Web 界面

## 许可证

版权所有 (c) 2024-2026 BlindGuard Team

---

**注意**：本系统仅供辅助使用，不能完全替代盲杖或导盲犬等传统辅助工具。使用时请始终保持警惕，注意安全。
