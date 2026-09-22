# BlindGuard 智能导盲系统 V2.5

## 系统简介

BlindGuard 是一款基于计算机视觉与大语言模型（LLM）的智能导盲辅助系统，为视障人士提供实时环境感知、风险评估、自然语言语音预警和智能对话服务。

V2.5 将智能体升级为“本地意图路由 + 有界工具编排 + 确定性安全守门员”架构：13 个工具由统一注册表管理 schema、权限、安全级别、超时和审计；能力目录声明 10 个角色（安全守门、实时感知、风险分析、无障碍导航、时序记忆、视觉观察、偏好管理、沟通、播报策略、运行监控），各自声明职责、权限边界和首选工具。每轮对话先经本地意图路由，最多预取 2 个本地只读工具，再由主责与协作角色（最多 3 个）各执行一个只读工具产出结构化证据，沟通结论模板化生成并经确定性安全守门员复核，最后才进入有缓存、有累计调用上限的 LLM 工具循环。原有通行安全边界、远程令牌、双模型生产管线和离线兜底保持不变。

## 系统架构

```
摄像头/视频
    │
    ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  检测引擎    │ → │ 红绿灯状态  │ → │  目标跟踪    │
│ Detection   │   │(HSV 识别)   │   │  Tracker    │
│ (主+辅模型) │   │             │   │ (趋势/轨迹) │
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
            ┌─────────────────────────────────────────────────────┐
            │   Agent Orchestrator（意图路由 + 有界角色编排）       │
            │ 安全守门 / 感知 / 风险 / 导航 / 记忆 / 视觉 / 偏好  │
            │ 沟通 / 播报策略 / 运行监控（能力目录共 10 个角色）    │
            └──────────────┬──────────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │ Tool Registry        │   │ Safety Guard         │
   │ 13 tools / schema    │   │ 红灯/黄灯/未知/高危  │
   │ 读写权限/超时/审计   │   │ 可覆盖 LLM，不授权通行│
   └──────────┬───────────┘   └──────────────────────┘
              ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │ 预取 + 角色工具执行   │ → │ LLM Function Calling │
   │ 有界证据读取与缓存    │   │ 权限策略/累计最多6次 │
   └──────────────────────┘   └──────────┬───────────┘
                                        ▼
   ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
   │ 时序事件记忆 │  │ 用户偏好记忆 │  │ 播报策略与历史   │
   │ 滑动窗口    │  │ JSON 持久化 │  │ 紧急度/冷却/原因 │
   └─────────────┘  └─────────────┘  └────────┬─────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │ VoiceAnnouncer   │
                                    │ 优先级队列/去重  │
                                    └──────────────────┘
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
├── requirements-ci.txt       # CI 轻量依赖锁定
├── constraints.txt           # 本机验证过的关键版本约束
├── tests/                    # 不加载 Torch 的单元测试
├── .github/workflows/ci.yml  # 编译与单元测试 CI
├── core/                     # 核心模块
│   ├── detection_engine.py   # 检测引擎（YOLO 封装）
│   ├── pipeline.py           # Web/评测共享生产管线（裁剪、双模型、风险）
│   ├── traffic_light.py      # 红绿灯状态识别（HSV 纯 CV）
│   ├── tracker.py            # 目标跟踪（IoU 匹配 + 接近/远离趋势）
│   ├── risk_evaluator.py     # 风险评估（多维度加权评分 + 红灯加权）
│   ├── scene_analyzer.py     # 场景分析（分布/空间/运动）
│   ├── voice_announcer.py    # 语音播报（优先级队列）
│   ├── config_manager.py     # 配置管理（YAML + .env 显式映射）
│   ├── model_validation.py   # 模型路径解析与启动前校验
│   ├── security.py           # 远程绑定与令牌认证
│   ├── safety_policy.py      # 通行安全护栏（禁止模型授权通行）
│   ├── agent_tools.py        # 工具注册表（schema/权限/超时/结果审计）
│   ├── agent_orchestrator.py # 意图路由、角色目录、工具循环、播报评分
│   ├── labels.py             # 统一类别中英文映射
│   ├── user_profile.py       # 用户偏好长期记忆（JSON 持久化）
│   └── agent.py              # 智能体兼容入口（编排 + 记忆 + 对话 + VLM）
├── templates/
│   └── index.html            # Web 界面（监控 + 语音对话）
├── docs/                     # 答辩与产品文档
│   ├── 演示脚本.md            # 90 秒演示流程与翻车预案
│   ├── 用户测试方案.md        # 志愿者内测方案与记录表
│   ├── 安全策略.md            # 确定性安全边界、失败策略与测试
│   ├── 评测复现与答辩证据.md  # 指标口径、复现命令与真实数据缺口
│   ├── 重训练指南.md          # 数据采集/标注/训练/验收/上线全流程
│   ├── pov_data_collection.md # 第一人称数据授权、脱敏与审核规范
│   └── git_history_cleanup.md # Git 大文件历史清理操作手册
├── scripts/                  # 只读审计工具（不自动重写/上传）
├── logs/                     # 日志目录
└── uploads/                  # 视频上传目录
```

## 核心功能

### 1. 实时目标检测（DetectionEngine）
- 主模型 `best_s.pt` 检测行人、自行车、汽车、摩托车、公交车、卡车、红绿灯
- 辅助模型 `best.pt` 只输出井盖、公共设施、盲道，按 IoU 去重合并，避免旧模型通用类误检
- 主模型支持 960 px 推理和竖屏中心裁剪，检测框自动回映全帧坐标
- 可配置置信度/IoU 阈值、设备、类别白名单和裁剪开关

### 2. 多维度风险评估（RiskEvaluator）
- **类别 × 距离 × 位置** 三维加权评分
- 五级风险：极高 / 高 / 中 / 低 / 安全
- 风险类别与阈值全部可在 config.yaml 配置

### 3. 场景分析（SceneAnalyzer）
- 目标分布统计、空间方位分析、前后帧运动趋势
- 场景类型判断：空旷/简单/复杂/拥挤/危险

### 4. 智能体层（BlindGuardAgent）
- **编排式专业架构**：能力目录声明安全守门员、实时感知员、风险分析员、无障碍导航员、时序记忆员、视觉观察员、偏好管家、沟通生成器、播报策略员、运行监控员共 10 个角色，各自声明职责、权限边界和首选工具，能力目录可通过 `get_agent_capabilities` 实时查询。每轮对话经本地意图路由后，仅由主责与协作角色（最多 3 个）各执行一个只读工具并写入 `role_contributions` 证据；沟通生成器与安全守门员不参与团队执行，而是在收尾阶段以固定台账形式追加沟通说明与确定性复核结论
- **统一工具注册表**：LLM 可调用 13 个工具，覆盖当前检测、场景概览、时序记忆、风险分析、非通行导航建议、画面观察、用户偏好、系统健康、能力目录、播报历史与播报策略；注册表统一做 JSON Schema 校验、读写权限、安全级别、超时控制、结果归一化和最近 100 次调用审计
- **本地意图路由与预取**：先识别偏好、能力、健康、历史、视觉、时序、环境七类意图，最多预取 2 个本地只读工具，再决定是否进入 LLM；外部 VLM 永不自动预取，避免延迟和隐私扩大
- **可解释工具循环**：默认最多 3 轮 function calling、累计最多 6 次模型工具调用；同轮成功只读结果进入缓存，写工具仅允许偏好意图，外部视觉工具仅允许明确视觉意图。状态接口返回主责角色、协作角色、结构化角色结论、预取/模型工具轨迹、`routing -> prefetch -> specialists -> llm -> tool_loop -> safety_review` 阶段、缓存命中、策略拒绝、成功率与耗时
- **播报策略与紧急度**：按风险、灯色、运动趋势、置信度和目标面积计算 0-100 紧急度；低风险/低置信度保持沉默，提供冷却、门槛、原因和保守静默说明，但静默绝不解释为环境安全
- **红绿灯状态识别**：纯 CV（HSV 颜色统计）判断红/黄/绿，零 API 成本；红灯自动加权风险并播报"红灯，请等待，不要通行"
- **确定性通行安全护栏**：红灯、黄灯、未知灯色、无可靠检测或高风险场景下，通行问题绕过 LLM 直接返回本地保守答复；绿灯也不代表通行许可，所有 LLM 通行答复都会在返回前二次拦截授权措辞
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
- 受保护页面/API/视频流令牌认证；前端动态内容使用 `textContent`，不注入动态 HTML

## 产品定位与安全边界

**价值主张**：与"用户举手机拍照"的被动式助盲工具（Seeing AI / Be My Eyes 等）不同，BlindGuard 是**主动式**的——系统持续感知环境，按风险等级主动语音预警，并针对中国道路定制了通用模型不认识的井盖、盲道类别。

**安全边界（重要）**：
- 本系统是**辅助感知工具，不是安全认证设备**：输出的是"提示"而非"指令"，不能替代盲杖、导盲犬或人的判断；
- 通行问题采用确定性硬约束：红灯、黄灯、未知灯色、无可靠检测或高风险目标存在时，不等待网络或 LLM，直接回答“等待/不能判断”；
- 绿灯只表示检测到绿色发光区域，**不代表系统批准通行**；绿色灯态仍须结合盲杖、交通规则与现场人工判断；
- LLM 只负责理解和描述环境，**没有通行授权能力**；其输出会经过本地后处理，出现“可以安全通行”“放心走”等授权措辞时整句替换为保守答复；
- 播报采取保守策略：低风险等级（low/safe）与低置信度目标不主动播报（可配置 `agent.announce.min_level` / `min_confidence`）；**不把漏检解释为安全，也不把低置信度结果作为通行依据**；
- 红绿灯灯色不确定时（逆光/遮挡）系统保持沉默，而非猜测。

**隐私原则**：目标检测与风险评估全部本地完成；仅当用户主动提问且配置了视觉端点时，才按需上传单帧画面给 VLM，未配置则该功能完全关闭。

**路线图**：① 当前形态（演示平台）→ ② 手机 App + 语音优先交互 + 视障志愿者内测（见 `docs/用户测试方案.md`）→ ③ 与无障碍生态合作（残联/厂商/公益）。

## 快速开始

### 1. 安装依赖
```bash
python --version     # 当前验证版本: Python 3.10.11
pip install -r requirements.txt
# 可选：使用项目验证过的关键版本约束
pip install -r requirements.txt -c constraints.txt
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
默认只监听 `127.0.0.1`。若改为 `0.0.0.0` 或其他非回环地址，必须设置至少 16 个字符的访问令牌，否则启动会被拒绝：

```powershell
$env:BLINDGUARD_SERVER_TOKEN = "请替换为随机长令牌"
python run.py
```

### 4. 访问界面
打开浏览器访问：http://localhost:5000

### 5. 自检与评测
```bash
python -m unittest discover -s tests -v   # 轻量单元测试，不加载 Torch
python smoke_test.py                  # 冒烟测试：模块/Agent/跟踪/红绿灯/偏好/接口全链路
python test_agent_tools.py            # 工具调用循环测试（本地模拟 LLM，无需真实 Key）

python evaluate.py perf               # 推理性能：各阶段耗时/FPS/CPU内存/播报延迟
python evaluate.py light --synthetic 60    # 红绿灯识别自检；有真实裁剪图用 --dir
python evaluate.py collect            # 采集人工抽检帧+红绿灯裁剪图
python evaluate.py bootstrap          # 从视频抽帧+伪标签，人工校正后生成测试集
python evaluate.py accuracy --data data_eval.yaml   # mAP / 各类别 AP
# 结果写入 evaluation/metrics_report.md，答辩可直接引用
```

> `evaluation/metrics_report.md` 中的 CPU 10.4 FPS 是早期单模型历史基线，不可直接代表当前
> `best_s.pt + best.pt`、960 px、竖屏裁剪配置。每次性能评测都会记录模型 SHA256、设备、
> 尺寸、阈值和裁剪设置；请在目标机器上重新运行 `evaluate.py perf` 后引用新结果。
> 2026-09-19 在 RTX 5060 Laptop GPU 上按当前双模型配置复测 300 帧为 **28.2 FPS**
>（管线中位 22.5 ms、P95 27.5 ms）；该数字是硬件相关性能，不等同于检测准确率。
> 模型重训流程见 `docs/重训练指南.md`，第一人称数据须按
> `docs/pov_data_collection.md` 完成授权、脱敏和场景隔离。

## 配置说明（config.yaml）

```yaml
detector:            # 检测引擎
  model_path: "best_s.pt"
  confidence: 0.5
  iou_threshold: 0.45
  img_size: 960      # 主模型推理尺寸
  portrait_crop: true
  aux_model_path: "best.pt"
  aux_img_size: 640
  aux_classes: ["manhole_cover", "public_facility", "tactile_paving"]

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
    min_level: "medium"
    min_confidence: 0.45
  architecture:      # 编排式多能力 Agent
    version: "2.5.0"
    max_tool_rounds: 3          # LLM 工具调用最多 3 轮
    max_model_tool_calls: 6     # 单轮模型工具调用总上限
    tool_trace_size: 80         # 有界工具轨迹
    event_history_size: 40      # 事件流上限
    announcement_history_size: 20
    enable_intent_planner: true # 本地意图路由
    enable_tool_prefetch: true  # 有界只读工具预取
    enable_specialist_team: true # 主责/协作角色执行并产出结构化证据
    max_specialist_roles: 3     # 每轮最多执行的主责与协作角色数量

server:
  host: "127.0.0.1"  # 非回环地址必须配置 auth_token
  port: 5000
  auth_token: ""     # 推荐用 BLINDGUARD_SERVER_TOKEN 环境变量
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
- `GET  /api/agent/status` - Agent 状态（角色目录 10 个、13 个工具、预取轨迹、LLM/视觉健康、记忆帧数）

### 访问认证

配置令牌后，首页、`/api/*` 和 `/video_feed` 都受保护。网页内请求会自动携带
`X-BlindGuard-Token`；命令行调用可使用专用 Header、Bearer 或仅对 MJPEG 使用 query token：

```bash
curl -H "X-BlindGuard-Token: $BLINDGUARD_SERVER_TOKEN" http://127.0.0.1:5000/api/system/status
curl -H "Authorization: Bearer $BLINDGUARD_SERVER_TOKEN" http://127.0.0.1:5000/api/agent/status
```

## 风险等级说明

| 等级 | 颜色 | 说明 | 语音优先级 |
|------|------|------|-----------|
| 极高风险 | 🔴 红色闪烁 | 高危目标非常近 | 插队播报 |
| 高风险 | 🟠 橙红 | 需立即避让 | 高 |
| 中等风险 | 🟡 橙色 | 注意观察 | 中 |
| 低风险 | 🔵 蓝色 | 留意周围 | 低 |
| 安全 | 🟢 绿色 | 当前未发现高风险目标，不代表确认可通行 | - |

## 版本历史

### V2.5.0 (2026)
- 专业角色从“声明职责”升级为真实执行：每轮由主责与协作角色（最多 3 个）各执行一个只读工具并产出结构化结论，运行记录写入 `role_contributions`
- 新增强制的角色执行阶段 `specialists`，完整轨迹为 `routing -> prefetch -> specialists -> llm -> tool_loop -> safety_review`
- 模型工具调用增加累计上限、写权限意图约束和视觉工具意图约束；未知工具与越权调用返回结构化拒绝证据
- 同轮成功只读工具结果进入缓存，模型重复读取时标记 `cache_hit=true`，避免重复执行
- `max_specialist_roles`、`max_model_tool_calls` 和 `specialist_execution_enabled` 进入配置与状态接口
- 新增角色执行、越权写偏好拦截、非视觉意图外呼 VLM 拦截、缓存与阶段轨迹回归测试

### V2.4.0 (2026)
- 智能体升级为“编排式专业能力 + 确定性安全守门员”：10 个角色声明职责、权限边界与首选工具，角色目录可通过状态接口和能力工具查询
- 工具注册表统一 13 个工具的 schema 校验、读写权限、安全级别、超时、结果归一化与有界调用审计
- 新增无障碍导航建议工具：输出停步、减速、盲杖确认、方位风险和非通行建议，明确不包含过街授权
- 新增播报策略工具：解释播报门槛、紧急度、冷却、保守静默与历史记录，静默不等同于环境安全
- 本地意图路由扩展为偏好、能力、健康、播报历史、视觉、时序、环境七类路径；最多预取 2 个本地只读工具，VLM 不自动预取
- 状态接口增加角色目录、工具预取开关、最近预取/模型工具轨迹、阶段化运行路径和真实安全改写证据；原始回复、最终回复、是否重写及是否仍含授权措辞分别记录
- 用户偏好写入保持白名单和范围双校验，拒绝未知偏好键

### V2.3.0 (2026)
- 新增确定性通行安全护栏：红灯/黄灯/未知灯色/无可靠检测/高风险问题绕过 LLM，绿灯不授权通行，主动播报和 LLM 回复统一做危险措辞拦截
- Web 应用与 `evaluate.py perf` 共用 `DetectionPipeline`，消除裁剪、双模型合并、跟踪和风险顺序漂移
- 新增远程绑定安全门槛：非回环地址无令牌时拒绝启动；首页、API 和 MJPEG 支持令牌认证
- 前端请求统一走 `apiFetch()`；动态状态改用 `DOM/textContent`，Service Worker 不缓存受保护资源
- 启动前校验主/辅助模型文件；性能与精度报告记录模型 SHA256、设备、尺寸、阈值和裁剪配置
- 新增无 Torch 依赖的单元测试、GitHub Actions CI、关键依赖约束和 Windows 换行/二进制属性
- 新增 Git 历史清理安全手册、只读审计脚本，以及第一人称数据授权/脱敏/清单审核脚手架

### V2.2.3 (2026)
- **yolov10s 容量升级上线**：BDD 路域 51.6 → **55.7**（traffic_light 47.5→55.8 +8.3、bus +6.9、
  person +5.5、car +4.3；唯一回退 motorcycle -7.0，恰为 BDD 样本最少类）；推理 14ms/帧不慢于 n 系
- 灯色识别器 HSV 网格调参（BDD 真实数据，调参/测试分离）：73.5% → **78.2%**，保守沉默 15.1%→9.0%
- 第一人称 POV 伪标注实验（负结果入库）：Mixkit+Coverr 727 张路域伪标注，路域 51.6→48.9——
  千张以下伪标注无增益、域外噪声反噬，数据引擎流水线已就绪待规模化
- 延长轮次实验（负结果入库）：同数据续训 6 轮，路域 51.1 vs 51.6 持平——堆轮数无效
- 教训入库：batch8@960 溢出共享内存慢 18 倍（8GB 卡用 batch6）；python 子进程需显式
  Git Bash OpenSSL 版 curl

### V2.2.2 (2026)
- 轨迹置信度：前端展示升级为目标历史最高确认度（实测行人单帧 70%→轨迹 81.2%/峰值 93.1%）
- v5 模型：接入 BDD100K 1 万张行车视角路域数据（18.6 万标注框），部署域 mAP50 39.9→**51.6**（+11.7），
  traffic_light 27.1→47.5（+20.4）、truck +21.6——路域数据对部署场景数量级改善
- 灯色识别首次真实验证：1,884 个真实红绿灯准确率 73.5%（保守沉默 15.1%）
- 竖屏帧中心裁剪推理（等效分辨率 +49%）；PYTORCH 显存碎片化修复；BDD 工具链入库

### V2.2.1 (2026)
- v4 模型：960px 高分辨率微调，mAP50 58.3 → **63.3**（traffic_light 43.4→51.8、car 58.7→65.2、bicycle 44.5→52.0）
- 置信度阈值校准 0.6→0.5（v3 阈值扫描：P91.5%/R41.0%，在漏检风险与误报干扰之间重新平衡；漏检仍不代表安全）
- 已知极限记录：竖屏素材 letterbox 压缩导致远处小灯仍难检出（实拍横屏不受影响）

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
- 依据抽检证据将置信度阈值 0.5 → 0.6 以降低当时的高误检，并确立"第一人称实拍数据重训"改进路线
- 评测报告完整化：性能 / 红绿灯 / 人工抽检三节，含方法学与改进清单

### V2.1.1 (2026)
- 播报分级保守策略：低于 medium 等级或置信度阈值的目标不主动播报，且不把未播报解释为安全
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
