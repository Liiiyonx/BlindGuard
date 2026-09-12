# BlindGuard 智能导盲系统 V2.1

## 系统简介

BlindGuard 是一款基于计算机视觉与大语言模型（LLM）的智能导盲辅助系统，为视障人士提供实时环境感知、风险评估、自然语言语音预警和智能对话服务。

V2.1 在 V2.0 的基础上把智能体层升级为**真正的 Agent**：LLM 可自主调用工具（查检测、查记忆、看画面、读写用户偏好），结合红绿灯状态与目标运动趋势进行时序推理，实现"感知-推理-行动"闭环。

## 系统架构

```
摄像头/视频
    │
    ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  检测引擎    │ → │ 红绿灯状态  │ → │  目标跟踪    │
│ Detection   │   │(HSV 识别)   │   │  Tracker    │
│ (YOLO/best) │   │             │   │ (趋势/轨迹) │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       ▼                 ▼                 ▼
┌─────────────────────────────────────────────────┐
│  风险评估器 RiskEvaluator（类别×距离×位置×灯色）  │
└───────────────────────┬─────────────────────────┘
                        ▼
                ┌─────────────┐
                │  场景分析器  │
                │SceneAnalyzer│
                └──────┬──────┘
                       ▼
            ┌───────────────────┐     ┌─────────────┐
            │   智能体核心       │ ←──→│  时序记忆    │
            │ Agent(LLM+工具)   │     │ (滑动窗口)   │
            │ 查检测/查记忆/看图 │     └─────────────┘
            └────────┬──────────┘     ┌─────────────┐
                     │       ←──────→ │ 用户偏好记忆 │
                     ▼                └─────────────┘
            ┌─────────────┐
            │  语音播报器  │
            │VoiceAnnouncer│
            │ (优先级队列) │
            └─────────────┘
```

检测在独立管线线程中单实例运行，多浏览器客户端共享同一份最新标注帧，FPS 与客户端数量无关。

```
BlindGuard/
├── app.py                    # 主应用（装配与调度各核心模块）
├── run.py                    # 启动脚本（含环境检查）
├── smoke_test.py             # 冒烟测试脚本
├── test_agent_tools.py       # 工具调用循环端到端测试（本地模拟 LLM）
├── evaluate.py               # 答辩指标评测（perf/accuracy/light/bootstrap/collect）
├── train.py                  # 模型重训练脚本（数据集yaml驱动，训练后自动评测对比）
├── config.yaml               # 唯一配置文件（检测/风险/语音/Agent/视觉）
├── .env                      # 私密配置（API Key，已 gitignore）
├── requirements.txt          # 依赖清单
├── core/                     # 核心模块
│   ├── detection_engine.py   # 检测引擎（YOLO 封装）
│   ├── traffic_light.py      # 红绿灯状态识别（HSV 纯 CV）
│   ├── tracker.py            # 目标跟踪（IoU 匹配 + 接近/远离趋势）
│   ├── risk_evaluator.py     # 风险评估（多维度加权评分 + 红灯加权）
│   ├── scene_analyzer.py     # 场景分析（分布/空间/运动）
│   ├── voice_announcer.py    # 语音播报（优先级队列）
│   ├── config_manager.py     # 配置管理（YAML + .env 显式映射）
│   ├── labels.py             # 统一类别中英文映射
│   ├── user_profile.py       # 用户偏好长期记忆（JSON 持久化）
│   └── agent.py              # 智能体（LLM 工具调用 + 记忆 + 对话 + VLM）
├── templates/
│   └── index.html            # Web 界面（监控 + 语音对话）
├── docs/                     # 答辩与产品文档
│   ├── 演示脚本.md            # 90 秒演示流程与翻车预案
│   ├── 用户测试方案.md        # 志愿者内测方案与记录表
│   └── 重训练指南.md          # 数据采集/标注/训练/验收/上线全流程
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

### 4. 智能体层（BlindGuardAgent）⭐ V2.1 升级
- **工具调用（Function Calling）**：LLM 可自主调用 6 个工具——查当前检测、查场景概览、查时序记忆、看画面（VLM）、读/写用户偏好——形成"感知-推理-行动"闭环，能回答"现在能过马路吗"这类需要组合推理的问题
- **红绿灯状态识别**：纯 CV（HSV 颜色统计）判断红/黄/绿，零 API 成本；红灯自动加权风险并播报"红灯，请等待绿灯再通行"
- **目标跟踪与运动趋势**：IoU 跟踪器为每个目标分配稳定 ID，基于面积变化判断接近/远离——播报从"前方有车"升级为"左侧汽车正在接近"
- **多模态看图（可选）**：接入 OpenAI 兼容视觉端点（Qwen-VL / GLM-4V 等），Agent 需要时"亲眼看"画面，识别文字、标志与 YOLO 类别之外的异常
- **用户偏好长期记忆**：对话中说"说慢一点"即可调整语速，JSON 持久化、跨重启生效、自动应用到 TTS
- **智能对话**：多轮上下文，前端支持语音输入（浏览器 Web Speech API）
- **离线兜底**：未配置 API Key 或调用失败时自动回退到模板播报与本地对话；LLM 连续失败自动熔断（暂停 30 秒），不阻塞系统
- 支持任意 OpenAI 兼容端点（DeepSeek / 通义千问 / 智谱 GLM / OpenAI）

### 5. 语音播报（VoiceAnnouncer）
- 优先级队列：极高风险插队播报
- 冷却去重 + 异步处理，不阻塞检测
- LLM 播报异步生成，不阻塞视频流

### 6. Web 可视化界面
- 实时视频流、FPS、检测目标数
- 整体风险等级（后端综合评估）、场景类型
- 智能对话面板（含 LLM 在线状态徽章）

## 产品定位与安全边界

**价值主张**：与"用户举手机拍照"的被动式助盲工具（Seeing AI / Be My Eyes 等）不同，BlindGuard 是**主动式**的——系统持续感知环境，按风险等级主动语音预警，并针对中国道路定制了通用模型不认识的井盖、盲道类别。

**安全边界（重要）**：
- 本系统是**辅助感知工具，不是安全认证设备**：输出的是"提示"而非"指令"，不能替代盲杖、导盲犬或人的判断；
- 播报采取保守策略：低风险等级（low/safe）与低置信度目标不主动播报（可配置 `agent.announce.min_level` / `min_confidence`），**宁可漏报，不误报**；
- 红绿灯灯色不确定时（逆光/遮挡）系统保持沉默，而非猜测。

**隐私原则**：目标检测与风险评估全部本地完成；仅当用户主动提问且配置了视觉端点时，才按需上传单帧画面给 VLM，未配置则该功能完全关闭。

**路线图**：① 当前形态（演示平台）→ ② 手机 App + 语音优先交互 + 视障志愿者内测（见 `docs/用户测试方案.md`）→ ③ 与无障碍生态合作（残联/厂商/公益）。

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

### 5. 自检与评测
```bash
python smoke_test.py                  # 冒烟测试：模块/Agent/跟踪/红绿灯/偏好/接口全链路
python test_agent_tools.py            # 工具调用循环测试（本地模拟 LLM，无需真实 Key）

python evaluate.py perf               # 推理性能：各阶段耗时/FPS/CPU内存/播报延迟
python evaluate.py light --synthetic 60    # 红绿灯识别自检；有真实裁剪图用 --dir
python evaluate.py collect            # 采集人工抽检帧+红绿灯裁剪图
python evaluate.py bootstrap          # 从视频抽帧+伪标签，人工校正后生成测试集
python evaluate.py accuracy --data data_eval.yaml   # mAP / 各类别 AP
# 结果写入 evaluation/metrics_report.md，答辩可直接引用
```

> **已完成的评测结论**（详见 `evaluation/metrics_report.md`）：CPU 推理 10.4 FPS、检测占耗时 90%；
> 人工抽检发现录屏素材与训练数据存在明显域差距（行人误检为 car、前景手套误检为 bicycle），
> 据此已将置信度阈值上调至 0.6，并列出以第一人称实拍数据重训的改进路线。
> **模型重训完整流程见 `docs/重训练指南.md`**（新数据集需补"行人"类，这是当前最大的能力缺口）。

## 配置说明（config.yaml）

```yaml
detector:            # 检测引擎
  model_path: "best.pt"
  confidence: 0.5
  iou_threshold: 0.45
  img_size: 640      # 推理图像尺寸

risk_engine:         # 风险评估
  area_thresholds:   # 距离阈值（面积占比），与距离分档一一对应
    very_close: 0.15
    close: 0.10
    medium: 0.05
    far: 0.01
  high_risk_classes: # 高危类别（含 manhole_cover 等自定义类）
    - "car"
    - "manhole_cover"

agent:               # 智能体
  enabled: true
  llm:
    base_url: "https://api.deepseek.com/v1"
    api_key: ""      # 留空！密钥放 .env
    model: "deepseek-chat"
  vision:            # 可选：多模态视觉端点（Agent 看图工具）
    base_url: ""     # 如 https://open.bigmodel.cn/api/paas/v4
    api_key: ""      # 或环境变量 BLINDGUARD_AGENT_VISION_API_KEY
    model: ""        # 如 glm-4v-flash / qwen-vl-plus
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
- `GET  /api/detections` - 当前检测结果（含风险等级、风险分值、红绿灯状态、运动趋势）
- `POST /api/chat` - 智能对话 `{"message": "前面有什么"}`（Agent 可自主调用工具）
- `GET  /api/agent/status` - Agent 状态（LLM 是否在线、工具数、视觉是否启用、记忆帧数）

## 风险等级说明

| 等级 | 颜色 | 说明 | 语音优先级 |
|------|------|------|-----------|
| 极高风险 | 🔴 红色闪烁 | 高危目标非常近 | 插队播报 |
| 高风险 | 🟠 橙红 | 需立即避让 | 高 |
| 中等风险 | 🟡 橙色 | 注意观察 | 中 |
| 低风险 | 🔵 蓝色 | 留意周围 | 低 |
| 安全 | 🟢 绿色 | 可正常通行 | - |

## 版本历史

### V2.2 (2026)
- 模型全面重训 v3：COCO train2017 全量 35,000 张真值、7 类（新增 motorcycle/bus/truck），
  mAP50 40.4 → **58.3**（person 63.5→73.2，car 39.9→58.7，bicycle +18.7，traffic_light +10.9）
- GPU 训练全流程落地：PyTorch cu128 离线安装、train2017 六段并行下载与断点补拉、
  分段虚拟拼接抽取（免 20GB 解压空间）
- 部署验证：行人+真车同时正确检出（旧模型该帧仅有人→car 假阳性）
- 旧 best.pt 转为定制类辅助模型（井盖/盲道/公共设施白名单），主辅 IoU 去重合并
- build_dataset_v3.py / fetch_remainder.py / train.py(--workers) 工具链入库

### V2.1.3 (2026)
- 发现并修复 PyTorch CPU 构建：离线安装 cu128 轮子，RTX 5060 首次可用于训练与推理
- 检测升级为双模型：主模型 best_v2.pt（COCO 真值 2926 张 GPU 微调，person AP50=63.5）
  负责通用类，原 best.pt 白名单仅输出定制类（井盖/盲道/公共设施），主优先 IoU 去重
- 行人误检为 car 的核心缺陷修复（抽检帧 2085：car 0.57 → person 0.9）
- 推理提速：RTX 5060 双模型 28.8ms/帧（≈35 FPS），此前 CPU 单模型 75ms
- build_dataset.py（COCO 转换/井盖定位/伪标注）与 train.py（--workers/--resume 适配 Windows）
- 修复 ConfigManager 丢弃 detector 扩展键、train.py numpy 打印 bug

### V2.1.2 (2026)
- evaluate.py 新增 collect 模式（人工抽检帧 + 红绿灯裁剪图采集）
- 修复检测输出中的 numpy 类型（无法 JSON 序列化）
- 首轮人工抽检（12 帧/15 框）发现录屏素材域差距：行人误检为 car、前景手套误检为 bicycle
- 依据抽检证据将置信度阈值 0.5 → 0.6（宁漏勿误），确立"第一人称实拍数据重训"改进路线
- 评测报告完整化：性能 / 红绿灯 / 人工抽检三节，含方法学与改进清单

### V2.1.1 (2026)
- 播报分级保守策略：低于 medium 等级或置信度阈值的目标不主动播报（宁可漏报不误报）
- 端到端播报延迟埋点，暴露于 /api/agent/status
- 新增 evaluate.py 评测脚本：推理性能/延迟、模型 mAP、红绿灯识别准确率、测试集 bootstrap
- 新增 docs/：演示脚本（含翻车预案）、用户测试方案
- README 新增产品定位与安全边界章节

### V2.1 (2026)
- 智能体升级为真正的 Agent：Function Calling 工具调用（查检测/查场景/查记忆/看图/读写偏好）
- 新增红绿灯状态识别（HSV 纯 CV）、目标跟踪与运动趋势（IoU 匹配）
- 新增多模态看图工具（可选接入 Qwen-VL / GLM-4V 等视觉端点）
- 新增用户偏好长期记忆（对话调整语速/播报详略，跨重启生效）
- 前端新增语音输入（Web Speech API）、红绿灯状态与趋势标签
- 修复：风险权重未归一化导致"极高风险"几乎无法触发
- 修复：帧尺寸硬编码 640×480，非 640 宽视频方位判断全部算错
- 修复：ConfigManager 环境变量解析错位、save() 往 YAML 写 JSON、img_size 配置不生效
- 性能：检测重构为单实例管线线程，多客户端共享视频流，FPS 与客户端数无关
- 架构：统一类别中文映射（4 处重复 → 1 处）、统一配置体系、剪除无用依赖
- LLM 连续失败自动熔断，端点不支持 tools 自动降级

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
