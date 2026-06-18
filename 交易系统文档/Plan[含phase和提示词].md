# QuantSaaS 从零复刻完整构建 Plan

> **阅读说明**
> 这是一份完整的系统构建指南。把每个 Phase 的 Prompt 直接粘贴进 Cursor（或你的 AI 编辑器），按顺序执行即可。

---

## 一、系统定位与核心架构

### 系统是什么

一款面向量化交易投资者的**全天候智能量化管理工具**：通过华尔街级风控状态机、遗传算法参数寻优，将复杂的动态策略降维成普通人能一键托管的 SaaS 财富水库。

### 三端物理部署

- **`saas`（云端）**：决策大脑，执行 `Step()`，下发交易指令，不持有任何 API Key
- **`agent`（用户本地）**：执行手，只负责调用券商接口下单并上报结果，不含任何策略代码
- **`lab`（本地算力机）**：实验室，专跑 GA 进化与回测，连同一个 Postgres 实例，不下发真实交易

### 全局技术决策（不可推翻的铁律）

**在开始前，把这六条铁律贴在显眼的地方。违反任何一条都要立即停下来。**

1. **策略同构**：回测与实盘必须调用同一个 `Step()` 实现，内部禁止出现 `if isBacktest` 分支
2. **策略纯函数**：`Step()` 内部禁止定时器、网络请求、数据库读写、任何文件 I/O
3. **API Key 物理隔离**：券商 API 凭证只存在于 `config.agent.yaml`，永不进入 SaaS 侧
4. **GORM Code-First**：数据库 schema 真源是 Go struct，只用 AutoMigrate，永不写 SQL migration 文件
5. **无量纲计算**：价格相关计算使用对数收益率或比率，禁止跨标的比较绝对价格
6. **单一 Postgres**：不分库，Redis 仅做缓存，不作信号传递通道

---

## 二、Sigmoid 动态天平 — 微观引擎设计哲学

> 这是系统的核心创新，其余策略机制需自行设计填充。

### 本质：仓位弹簧系统

Sigmoid 动态天平用 Sigmoid 函数实时计算**目标持仓权重**，通过买卖浮动仓使实际权重趋近目标权重。它是一个与信号来源无关的通用框架——你可以把任何归一化的市场信号接入它。

### 核心公式：Sigmoid 目标权重

```
CurrentWeight  = FloatHold × Price / TotalEquity

Signal         = 【你的市场信号，任意归一化标量，正值倾向减仓，负值倾向加仓】

EffectiveBeta  = max(0.01,  β × MarketBetaMultiplier)
InventoryBias  = clamp(CurrentWeight, 0, 1) − 0.5
Exponent       = EffectiveBeta × Signal + γ × InventoryBias

TargetWeight   = 1 / (1 + e^Exponent),  clamp(0, 1)
```

### 公式解读

| 场景 | Signal | Exponent 方向 | TargetWeight | 动作 |
|---|---|---|---|---|
| 你的信号看空 | 正 | 增大 | < 0.5 | 减仓 |
| 你的信号看多 | 负 | 减小 | > 0.5 | 加仓 |
| 仓位 > 0.5，γ > 0 | 任意 | 额外增大 | 进一步压低 | 均值回归 |
| 仓位 < 0.5，γ > 0 | 任意 | 额外减小 | 进一步拉高 | 均值回归 |

- **Signal**：你的策略信号，可以是均值回归信号、动量信号、突破信号，甚至多信号的加权合成——只要归一化为一个标量即可接入
- **β（激进系数）**：越大调仓越频繁，市场状态感知层可在极端行情时动态放大 β
- **γ（仓位偏置系数）**：`γ=0` 时纯信号驱动，`γ>0` 时叠加均值回归力防止仓位极端漂移
- **MarketBetaMultiplier**：来自上层市场状态感知，行情极端时放大 β 自动加速响应

### GA 在这里做什么

Signal 通常由多个因子线性合成：

```
Signal = a × X1 + b × X2 + c × X3 + ...
```

其中 X1、X2、X3 是你选择的市场特征（无量纲化后的标量），a、b、c 是对应的权重系数。这些系数就是**染色体的一部分**，由遗传算法在历史数据上搜索最优值。

举例：如果你设计了三个无量纲特征 X1（价格偏离类）、X2（动量类）、X3（加速度/突破类），合成信号就是 `a×X1 + b×X2 + c×X3`，GA 搜索的就是让这个合成信号喂进 Sigmoid 后，在多时段历史回测中跑赢被动 DCA 基准的最优 (a, b, c)。

你选什么特征完全由你的策略逻辑决定——均值回归、动量、突破、成交量异动、北向资金/主力资金流向……GA 的搜索机制是完全通用的，它不关心 X 是什么，只负责找到让系统在历史上表现最好的系数组合。连同 β、γ 等 Sigmoid 本身的参数一起，整个微观引擎的参数空间就是一个高维搜索问题，GA 是求解它的工具。所以为了防止过拟合，你需要在回测中自行加入乱序回测或者蒙特卡洛算法等内容。因此类内容对性能的影响过于大，本次plan没有写入。

### 理论订单与楔形区过滤

```
DeltaWeight    = TargetWeight − CurrentWeight
TheoreticalCNY = DeltaWeight × TotalEquity
```

粉尘拦截规则（防止无效小额交易）：
- `|TheoreticalCNY| ≥ 最小阈值`：直接下单
- `|TheoreticalCNY| ∈ (0, 最小阈值)`：仅在非安静态 **且** 满足楔形突破条件时强制最小订单，否则归零
- 楔形突破条件：`|DeltaWeight| ≥ 阈值` 或 `VolatilityRatio ≥ 阈值`
- `VolatilityRatio = clip(MAV短期 / MAV长期, 0.1, 3.0)`，MAV 为平均绝对涨跌（非 ATR）

---
以下是具体的vibe coding分步骤指令
## Phase 0 — 环境初始化与 AI 协作基础设施

### 目标
在项目根目录建立 AI 工作约束文件（CLAUDE.md/AGENTS/.cursoruls），让 Cursor 在每次对话中自动加载项目规范。初始化 Go 项目依赖。

### Context
这一步决定了后续所有 AI 对话的质量。CLAUDE.md 是 AI 的"宪法"，每次对话开始时自动读入。

### Prompt
```
帮我完成以下两件事：

第一，在项目根目录创建各种 约束文件 文件，内容包括以下几部分：

"唯一功能真源"部分：声明当前功能只依据 docs/ 下的三份文档（系统总体拓扑结构、策略数学引擎、进化计算引擎），三份文档没有定义的功能不进入实现。

"工作顺序"部分：列出四条规则——涉及策略和回测先读对应文档；涉及 Go 后端遵守 GORM Code-First 只用 AutoMigrate；涉及价格计算优先无量纲表达；涉及架构边界保持 SaaS-Strategy-Agent 分工不做预防性解耦。

"核心约束"部分：列出五条铁律——策略必须满足复利前置条件；回测与实盘调用同一 Step() 实现；Step() 只在 SaaS 侧执行；策略包内部禁止网络数据库文件 I/O；API Key 只能在 config.agent.yaml。

"代码目录"部分：列出 cmd/saas/ cmd/agent/ internal/saas/ internal/agent/ internal/strategy/ internal/strategies/[策略名]/ internal/quant/ internal/adapters/backtest/ 的各自职责说明。

"验证命令"部分：go list ./... 和 go test ./...

第二，初始化 Go 项目，并安装以下依赖：gin、gorm + postgres driver、go-redis、golang-jwt、robfig/cron、zap、gorilla/websocket、testify。
第三，为整个项目构建一些基础的SKILLS，至少包含系统架构师、量化交易数学专家、go后端专家、部署与运维专家
```

### 预期产出
- `CLAUDE.md`
- `go.mod` + `go.sum`
- 基础目录骨架
- AGENT SKILL

---

## Phase 1 — 三份真源文档（需要自己填写内容）

### 目标
在写任何代码之前，用文档把系统的设计意图固化。这三份文档是整个系统的"法律"，是后续所有代码的唯一依据。其中进化计算引擎已经有原始参考文件，直接参考即可。

> **重要说明**：下面给出的是三份文档的**结构骨架**，每个标题下的内容需要你自己填写。这是你的策略设计空间，没有标准答案。可以用自然语言与AI对话，描述你要的需求，让AI补齐内容。

### 1A. 系统总体拓扑结构文档

### Prompt
```
帮我在 docs/ 目录下创建"系统总体拓扑结构.md"。这份文档定义系统有哪些物理端、有哪些逻辑模块、状态如何在它们之间流转，以及系统的生命周期动作。不含任何具体策略公式。

文档结构如下，每个章节标题下用三行以上的文字描述清楚这部分的设计决策：

第 0 章：架构哲学

第 1 章：三端物理部署形态（saas 云端 / agent 用户本地 / lab 算力机的各自职责与禁区）

第 2 章：app_role 三态行为矩阵（saas/lab/dev 各自开放和限制哪些能力，用表格表示）

第 3 章：逻辑模块与职责边界（Strategy 策略模块 / Instance 实例模块 / Evolution 进化模块 / Auth 认证模块，明确每个模块的职责边界和禁区）

第 4 章：全局状态总线（单一 Postgres + Redis 仅缓存；列一张数据所有权表，每类数据的真源在哪一端）

第 5 章：WebSocket 通信协议（消息类型全表，含 auth/heartbeat/command/command_ack/delta_report/report_ack 的方向与触发时机；TradeCommand 字段语义；DeltaReport 字段语义；状态收敛与天然自愈机制）

第 6 章：系统级生命周期动作（系统初始化流程 / Cron Tick 驱动 Step() 的完整流程 / Agent 断线重连指数退避 / 优雅停机与状态快照）

第 7 章：不可推翻的技术决策（把前述六条铁律以及你自己认为不可推翻的其他约束完整列出）
```

### 1B. 策略数学引擎文档

### Prompt
```
帮我在 docs/ 目录下创建"策略数学引擎.md"。这份文档是策略逻辑的数学规格书，定义 Step() 函数的完整输入输出契约，以及内部各层的计算逻辑。

文档结构如下：

第 0 章：引擎身份（纯函数无状态转换器；唯一入口 Step(StrategyInput)→StrategyOutput；铁律复述）

第 1 章：资产结构三态（Portfolio State）
——DeadHold（宏观底仓，只进不出）/ FloatHold（微观浮动仓，可买卖）/ ColdSealedHold（冷封存，永不释放）
——TotalEquity 公式、ReserveFloor 公式、SpendableCNY 公式、CurrentMicroWeight 公式（这四个是通用架构，保留）
——micro_reserve_pct 参数语义与默认值（可进化）

第 2 章：市场状态感知层
【核心设计空间一：如何对市场状态分类？分几类？每类用什么特征判断？】
必须定义"各种态（牛/熊/安静）"的判断逻辑（安静态下微观粉尘订单归零）。

第 3 章：信号与目标函数框架（可选的 MPC 概念层描述）

第 4 章：宏观引擎
【宏观引擎如何决定何时买、买多少？】
需要你设计具体的DCA策略，宏观引擎关注的是长期的趋势，这一点请注意。

第 5 章：微观引擎
直接按本文档第二章"Sigmoid 动态天平"的公式和解释填入。
包含：信号公式、Sigmoid 目标权重公式、理论订单计算、楔形区过滤规则。

第 6 章：DeadHold 释放规则
定义一下再什么情况下，可以把宏观引擎买入的底仓份额（A 股或黄金 ETF/现货）转为浮动仓位（即可以卖出）

第 7 章：可进化参数契约（Chromosome）
【你的核心设计空间三：哪些参数交给 GA 去搜索？】
必须包含：字段清单（名称/类型/语义/默认值/边界）、硬边界约束、结构约束（如 EMA 顺序约束、时间周期相对论锁）
必须分清：哪些是染色体（参与代内交叉变异），哪些是出生点参数（Epoch 级冻结，不进入基因组）

```

### 1C. 进化计算引擎文档

### Prompt
```
帮我在 docs/ 目录下创建"进化计算引擎.md"。这份文档定义 GA 遗传算法引擎的完整规格——它是一个纯计算黑盒，不关心具体策略的内部结构。

文档结构如下：

第 0 章：核心定位（纯计算线程；通过抽象接口驱动种群生命周期；不 import 任何具体策略实现）

第 1 章：探索窗口定义
1.1 多时段坩埚（四个评估窗口：全量历史/5年/2年/6个月，适应度权重分别为 0.40/0.30/0.20/0.10；严禁未来数据泄露）
1.2 基因空间的三个语义操作：Sample（随机采样）/ Clamp（修复越界）/ Validate（验证合法）
1.3 出生点冻结（Epoch 级，全种群共享，不参与代内交叉变异）

第 2 章：进化生命周期动作
种群初始化策略（精英继承 10% + 强化变异 40% + 完全随机 50%；index 0 始终为当前种子冠军原样）
并发适应度评估（固定大小 worker pool，上限 = min(NumCPU, PopulationSize)）
锦标赛选择（TournamentSize=3，随机抽取取最优）
均匀交叉（Uniform Crossover，每维度 50% 概率独立从两父代选取）
加性高斯变异（每维度独立 Bernoulli 概率决定是否变异，变异量为正态分布）
精英保留（Top N 个个体直接进入下一代）
收敛检测 + 变异斜坡（Mutation Ramp）：连续 N 代无改善则放大变异概率和幅度，上限触及且仍无改善才 Early Stop
基因组指纹缓存（FNV-1a-64 哈希，精度 1e-6，命中缓存则跳过重复回测评估）

第 3 章：适应度黑盒（Fitness Function）
多窗口坩埚分数公式：Alpha = ROI策略 - ROI被动DCA；SliceScore = Alpha - 1.5 × max(0, MaxDD策略 - MaxDD被动DCA)
Fatal 判断：MaxDD ≥ 88% 时 SliceScore = -99999（硬否决，立即返回）
加权汇总：ScoreTotal = 0.40×全量 + 0.30×5年 + 0.20×2年 + 0.10×6个月
Ghost DCA 基准定义（种子资本买入 + 每自然月月初注资全买，作为被动对照组）
Modified Dietz 收益率（剔除注资跳变对 NAV 的影响）
级联短路（按 6m→2y→5y→全量 顺序评估，fatal 立即退出跳过后续长窗口）

第 4 章：EvolvableStrategy 接口（8-verb 契约）
StrategyID / Sample / Mutate / Crossover / Fingerprint / Evaluate / DecodeElite / EncodeResult
说明每个动词的职责，强调引擎对染色体内部字段完全不可见

第 5 章：EvaluablePlan 只读上下文
Epoch 启动时构建，整个世代内不可变，包含：标的信息 / 出生点快照 / 四个坩埚窗口 / 预计算的 DCA 基线

第 6 章：结果交付与基因角色三态
challenger（进化产出，等待人工审批）→ champion（当前活跃冠军）→ retired（历史存档）
人工 Promote 流程（DB 事务：旧 champion→retired，challenger→champion；Redis 缓存立即失效）

第 7 章：HTTP 触发契约
POST /api/v1/evolution/tasks 的参数列表（pop_size, max_generations, spawn_mode）
GET /api/v1/evolution/tasks 的返回结构
```

---

## Phase 2 — 基础设施层（Config + DB + Auth）

### 目标
搭建系统的物理基础：配置加载、GORM 数据库模型定义（全量 AutoMigrate）、Redis 客户端、JWT 工具。

### Context
GORM Code-First 的核心意义：所有数据库结构变更都通过修改 Go struct 来完成，AutoMigrate 自动同步，完全不存在手写 SQL 文件的概念。这是整个项目的 schema 管理哲学。

### Prompt
```
请阅读 docs/系统总体拓扑结构.md，然后为我实现 Go 项目的基础设施层。

总体约束：模块使用 gin 框架，ORM 为 GORM + postgres driver，日志使用 zap，所有数据库模型只使用 GORM struct tag 定义，不写任何 SQL 文件。

请实现以下内容：

一、internal/saas/config/config.go
定义 Config 结构体（包含 AppRole / Database / Redis / JWT / Server 五个子配置），实现从 config.yaml 文件加载的函数。AppRole 取值为 "saas" / "lab" / "dev"。同时创建一份 config.yaml 模板，不含任何密钥，密钥字段留空并注明需通过环境变量注入。

二、internal/saas/store/models.go
用 GORM struct 定义所有核心数据模型：
- User（用户与订阅计划）
- StrategyTemplate（策略模板注册表，包含 ID/Name/Version/IsSpot 字段，以及 Manifest JSON blob）
- StrategyInstance（策略实例，包含状态字段 RUNNING/STOPPED/ERROR，以及关联的 Template 和 User）
- PortfolioState（实例账户快照，包含 CNYBalance/DeadHold/FloatHold/ColdSealedHold/TotalEquity/LastProcessedBarTime）
- RuntimeState（策略运行时状态，JSON blob 字段，由 Step() 产出后持久化）
- SpotLot（仓位 lot 记录，包含 LotType 字段取值 DEAD_STACK/FLOATING/COLD_SEALED，Amount/CostPrice/CreatedAt/IsColdSealed）
- TradeRecord（成交记录，包含 ClientOrderID/Action/Engine/Symbol/FilledQty/FilledPrice/Fee）
- SpotExecution（原始成交明细，pending→filled/failed 状态机）
- AuditLog（审计日志，EventType + Payload JSON blob）
- GeneRecord（基因库，包含 StrategyID/Role challenger-champion-retired/ParamPack JSON/ScoreTotal/MaxDrawdown）
- EvolutionTask（进化任务，包含 Status/Progress/Config JSON）
- KLine（历史 K 线，包含 Symbol/Interval/OpenTime/OHLCV 字段，在 Symbol+Interval+OpenTime 上建唯一索引）

三、internal/saas/store/db.go
实现 NewDB(cfg) 函数：建立 Postgres 连接，对以上所有模型执行 AutoMigrate，返回封装好的 DB 对象。

四、internal/saas/store/redis.go
实现 Redis 连接封装，提供 Get/Set/Del 三个基础方法。用途说明：冠军基因缓存（key: champion:{strategyID}）、会话缓存，不用于信号传递。

五、internal/saas/auth/service.go
实现 SignToken(userID uint, role string) 和 ParseToken(tokenStr string) 两个函数，使用 golang-jwt 库。
```

---

## Phase 3 — 量化数学基础层（internal/quant）

### 目标
实现所有策略共用的数学工具库：基础统计函数、资产三态管理、Ghost DCA 基准、以及微观引擎的 Sigmoid 动态天平。

### 3A. 基础数学工具

#### Prompt
```
请实现 internal/quant/ 目录下的数学基础工具，这些是所有策略共用的底层函数。

铁律：所有价格相关计算必须无量纲化，禁止在函数内部跨标的比较绝对价格。

math.go：实现以下无状态纯函数
- EMA：对任意 float64 序列计算指数平滑均线，输入序列 + 周期，返回最新一个值
- StdDev：对任意 float64 序列计算样本标准差，输入序列 + 周期，返回最新窗口的标准差
- MAVAbsChange：最近 L 根收盘的平均绝对涨跌（不是 ATR，不依赖 High/Low），公式为最近 L 根收盘两两绝对差值之和除以 L-1
- ClipFloat64：将 float64 值夹紧到 [lo, hi] 区间
- RoundToCNY：将 float64 四舍五入到两位小数

data.go：定义通用数据结构
- Bar：一根 K 线，包含 OpenTime/Open/High/Low/Close/Volume
- StrategyInput：策略输入快照，具体字段参考 docs/策略数学引擎.md 第 8 章
- StrategyOutput：策略输出意图集，具体字段参考 docs/策略数学引擎.md 第 8 章
- PortfolioSnapshot：账户快照，包含 CNYBalance/DeadHold/FloatHold/ColdSealedHold

closes.go：实现 ACL 降级工具函数
- ExtractCloses：从 []Bar 中提取收盘价序列 []float64
- ExtractTimestamps：从 []Bar 中提取时间戳序列 []int64
说明：现货类标的策略（A 股/黄金 ETF，`IsSpot=true`）的 OHLCV 降级必须在这里完成，策略内核代码禁止直接依赖 Bar 结构体
```

### 3B. 资产仓位管理

#### Prompt
```
请实现 internal/quant/lot.go，提供仓位 lot 的管理逻辑。

SpotLot 结构体包含：LotType（DEAD_STACK / FLOATING / COLD_SEALED 三态）、Amount、CostPrice、CreatedAt、IsColdSealed 布尔标志。

需要实现以下功能：
- 计算 DeadHold 总量（IsColdSealed=false 的 DEAD_STACK 的 Amount 之和）
- 计算 FloatHold 总量（FLOATING 的 Amount 之和）
- 软释放（Soft Release）：从 DEAD_STACK lot 中，筛选出老化超过指定月数的非 ColdSealed lot，按最大释放比例限制，将不超过"可卖出缺口"的数量转换为 FLOATING 类型，保留原始成本
- 硬释放（Hard Release）：当微观引擎产出卖出意图但 FloatHold 不足时，从 DEAD_STACK（非 ColdSealed，不限老化时间）中补足差额转为 FLOATING

铁律：ColdSealedHold 标记的 lot 任何情况下不可被释放。
```

### 3C. Sigmoid 动态天平（微观引擎）

#### Prompt
```
请实现 internal/quant/micro_engine.go，包含微观引擎的完整逻辑，核心是 Sigmoid 动态天平。

这是系统最核心的创新点：用 Sigmoid 函数实现动态目标仓位权重，同时支持均值回归、趋势跟踪、仓位反馈三种机制的叠加。实现时请严格遵循以下数学规格，不得改变公式结构：

输入结构体包含以下字段：
- 收盘价序列（[]float64）和当前价格
- 当前微观仓位权重（FloatHold × Price / TotalEquity）
- 总权益 TotalEquity
- 来自染色体的可进化参数：你认为适合你策略的信号参数、sigma_floor（σ 最小值）、beta（Sigmoid 激进系数）、gamma（仓位偏置系数）
- 来自市场状态感知层的参数：BetaMultiplier（动态放大系数）、IsQuiet（是否安静态）

输出结构体包含：TargetWeight / Signal（调试用）/ TheoreticalCNY / OrderCNY（经过滤后的实际订单金额）/ VolatilityRatio（调试用）

计算步骤如下，请严格按照顺序实现：

第一步：计算 EMA（窗长固定为不可进化常量，你在代码中定为常量 MicroSignalEMABars）和 σ（窗长固定为不可进化常量 MicroSignalStdDevBars）。σ 取 max(实际标准差, sigma_floor)。如果 σ=0 则跳过本次决策。

第二步：计算无量纲信号。可以是你定的任何计算方式。

第三步：计算 Sigmoid 目标权重。EffectiveBeta = max(0.01, beta × BetaMultiplier)。InventoryBias = clamp(CurrentWeight, 0, 1) - 0.5。Exponent = EffectiveBeta × Signal + gamma × InventoryBias。TargetWeight = 1/(1+exp(Exponent))，clamp 到 [0,1]。

第四步：计算理论订单。DeltaWeight = TargetWeight - CurrentWeight。TheoreticalCNY = DeltaWeight × TotalEquity。

第五步：计算 VolatilityRatio。使用固定常量 MicroVolRatioLongBars 和 MicroVolRatioShortBars 计算两个时间窗口的 MAVAbsChange，比值 clip 到 [0.1, 3.0]。数据不足时默认为 1.0。

第六步：楔形区过滤。|TheoreticalCNY| >= 最小订单阈值时直接以原值下单。在 (0, 最小订单阈值) 范围内，当非安静态且满足楔形突破条件（|DeltaWeight| 超过仓位变动阈值 OR VolatilityRatio 超过波动率阈值）时强制下最小订单（保持符号），否则 OrderCNY = 0。具体阈值数值在你的文档或染色体中定义。

请在函数注释中写入 Sigmoid 动态天平的设计哲学：Signal 是外力（市场信号），InventoryBias 是弹簧恢复力，Beta 是弹簧刚度，Gamma 决定是否启用弹簧，VolatilityRatio 楔形过滤控制安静期粉尘。
```

### 3D. 市场状态感知层

#### Prompt
```
请实现 internal/quant/market_state.go。

【这是你需要填充自己策略的地方，有很多经典的模型可以使用，比如马尔科夫之类的】

先阅读 docs/策略数学引擎.md 第 2 章，了解我对市场状态分类的设计。

MarketState 结构体必须包含以下字段（这是与宏观/微观引擎的接口契约，不可更改）：
- State string（你的状态枚举值）
- TimeDilationMultiplier float64（给宏观引擎，1.0 为正常，>1.0 扩展时间窗口）
- BetaMultiplier float64（给微观 Sigmoid，1.0 为正常，>1.0 加速调仓）
- IsQuiet bool（true 时微观粉尘订单归零）

ComputeMarketState 函数的输入和内部分类逻辑，请完全按照 docs/策略数学引擎.md 第 2 章的规格实现。
```

### 3E. 宏观引擎

#### Prompt
```
请实现 internal/quant/macro_engine.go。

【这是你需要填充自己策略的地方。宏观引擎负责长期建仓节奏，典型设计包括但不限于：固定周期定投、基于市场状态加速/减速的动态定投、价格偏离均值触发的加仓等。关键约束是只买不卖，且要有死线兜底机制防止资金长期闲置。】

先阅读 docs/策略数学引擎.md 第 4 章，了解你自己对宏观引擎的设计。

MacroDecisionInput 结构体需要包含的字段，以及 ComputeMacroDecision 函数的完整逻辑，请完全按照 docs/策略数学引擎.md 第 4 章的规格实现。

记住宏观引擎铁律：只产出 BUY 意图，绝对不产出 SELL 意图；订单金额需与 SpendableCNY 做 clamp；小于最小订单阈值 100 CNY 的订单不执行。
```

### 3F. Chromosome 可进化参数结构体

#### Prompt
```
请实现 internal/quant/genome.go，定义可进化参数的完整结构体、边界约束和辅助函数。

先阅读 docs/策略数学引擎.md 第 7 章，了解我的染色体设计。

需要实现：
一、Chromosome 结构体（字段、类型、json tag 完全按照文档第 7 章定义）

二、HardBounds 常量，定义每个字段的合法数值范围 [min, max]

三、ClampChromosome 函数：将所有字段 clamp 到合法范围，同时修复结构约束（EMA 顺序约束、相对论锁等），文档第 7 章有定义。变异后必须调用此函数。

四、DefaultSeedChromosome 变量：产品默认冠军种子值，作为 GA 冷启动时的初始个体和 JSON 解码失败时的回退值。具体值按文档定义填写。

五、SpawnPoint 结构体（出生点）：包含 Policy（资金政策，含月度注资/死线比例/释放阈值等）和 Risk（风险边界，含手续费率/全局止损等）。这部分参数不参与代内交叉变异，不进入基因组指纹。
```

### 3G. Ghost DCA 基准

#### Prompt
```
请实现 internal/quant/ghost_dca.go，提供被动 DCA 基准模拟器，供 GA 适应度评估时作为对照组使用。

GhostDCAConfig 包含初始资本和每月注资金额两个字段。

SimulateGhostDCA 函数逻辑：以第一根 bar 的收盘价用全部初始资本买入标的资产（如沪深300ETF或黄金ETF份额）；之后每个自然月月初，将 MonthlyInject CNY 全部用于加仓买入；全程记录 NAV 曲线用于计算最大回撤。

返回 GhostDCAResult，包含 FinalEquity / TotalInjected / MaxDrawdown / ROI 四个字段。

ROI 使用 Modified Dietz 方法计算：剔除注资跳变对 NAV 的影响。公式为：(期末权益 - 期初权益 - 现金流之和) / (期初权益 + Σ(现金流_i × 加权因子_i))，其中加权因子 = (总天数 - 注资发生日) / 总天数。

同时在此文件中实现 MaxDrawdown 计算函数：基于 NAV 曲线，计算峰值到谷底的最大相对回撤。
```

---

## Phase 4 — 策略模块（Step() 主函数）

### 目标
实现策略的主函数 `Step()`，这是整个系统的决策核心，回测和实盘共用同一个实现。

### Context
策略模块是一个纯函数包。整个包的唯一对外入口是 `Step()`。包内部不能有任何 I/O。所有 OHLCV 降级（把 Bar 序列变成 []float64）必须在调用 Step() 之前完成，策略内核只消费 []float64。

### Prompt
```
请阅读 docs/策略数学引擎.md，然后实现 internal/strategies/[策略名]/ 目录下的策略模块。

整个模块的文件结构建议如下：
- manifest.go：策略元数据（ID/Name/Version/IsSpot/Description）
- params.go：策略参数解析，包含从 ParamPack JSON 解析出 Chromosome + SpawnPoint 的函数
- state.go：RuntimeState 结构体，定义需要跨 tick 持久化的策略运行时状态字段（按文档中你的设计填写）
- macro.go：调用 quant.ComputeMacroDecision 的薄包装，注入文档规定的上下文参数
- micro.go：调用 quant.ComputeMicroDecisionV4 的薄包装
- dead_release.go：实现 DeadHold 软释放和硬释放的决策逻辑（调用 lot.go 的工具函数）
- step.go：主函数 Step(input quant.StrategyInput, params Params) quant.StrategyOutput，按以下顺序组装：
  1. 数据窗口充足性检查（不足则返回空输出）
  2. 从 input.Closes 和 Portfolio 快照计算 TotalEquity / SpendableCNY / CurrentMicroWeight
  3. 调用 ComputeMarketState 得到市场状态超参
  4. 调用宏观引擎得到宏观订单意图
  5. 调用微观引擎（Sigmoid 动态天平）得到微观订单意图
  6. 调用释放规则得到底仓释放意图
  7. 更新 RuntimeState（更新 LastProcessedBar 及你在文档中定义的其他持久化状态）
  8. 组装并返回 StrategyOutput

铁律检查清单（实现完成后逐一确认）：
- Step() 函数体内没有任何 http / sql / os / time.Now() 调用
- Step() 函数体内没有 if isBacktest 或类似分支
- 策略包的任何文件都没有 import 网络/数据库相关包
- 只使用 input.Closes []float64，没有直接使用 quant.Bar
```

---

## Phase 5 — 遗传算法进化引擎（GA Engine）

> **这是系统的算法核心，详细程度高于其他模块。**

### 架构关系

```
EvolutionEngine（调度器，不知道染色体字段名）
    ↓ 通过 EvolvableStrategy 8-verb 接口
[YourStrategy]Evolvable（策略侧适配器）
    ↓ 调用
RunBacktest（回测适配器）
    ↓ 调用
Step()（策略纯函数，与实盘相同）
```

### 5A. EvolvableStrategy 接口 + 策略侧实现

#### Prompt
```
请阅读 docs/进化计算引擎.md，然后实现 internal/saas/ga/evolvable.go 和 internal/saas/ga/[策略名]_evolvable.go。

evolvable.go 定义以下内容：
- Gene = any（不透明载体类型别名，引擎通过接口操作，不读取内部字段）
- DCABaseline 结构体（FinalEquity / TotalInjected / MaxDrawdown 三个字段，Epoch 启动时预计算 Ghost DCA 结果）
- EvaluablePlan 结构体（Symbol/TemplateName/Spawn/LotStep/LotMin/Windows/DCABaselines/AggregateCache，Epoch 启动时构建，整个世代内只读；其中 Symbol 为标的代码）
- EvolvableStrategy 接口，包含 8 个方法：StrategyID / Sample / Mutate / Crossover / Fingerprint / Evaluate / DecodeElite / EncodeResult

[策略名]_evolvable.go 为具体策略实现该接口，方法说明：

Sample：从染色体的合法边界内均匀随机采样一个 Chromosome，调用 ClampChromosome 修复结构约束后返回

Mutate：对每个染色体字段以独立的 Bernoulli 概率 prob 决定是否变异，变异量为 NormFloat64() × 该字段的步长 × scale，变异后调用 ClampChromosome

Crossover：对两个父代 Chromosome 的每个字段以 0.5 概率独立选择来源（均匀交叉），组装后调用 ClampChromosome 修复结构约束

Fingerprint：对所有染色体字段进行精度为 1e-6 的量化后，用 FNV-1a-64 哈希生成唯一指纹，相同参数（精度 1e-6 内）的染色体应产生相同哈希

Evaluate：执行多窗口坩埚评估。按 plan.Windows 升序（短→长）依次调用 RunBacktest，计算每个窗口的 Alpha 和 SliceScore，MaxDD >= 88% 时立即返回 fatal=-99999（级联短路），否则按 plan.Windows[i].Weight 加权汇总返回 ScoreTotal

DecodeElite：从 ParamPack JSON 中解码出 Chromosome，如果 raw 为空或解码失败则返回 DefaultSeedChromosome

EncodeResult：将冠军 Chromosome 和 SpawnPoint 序列化为 ParamPack JSON blob 存入数据库

注意包位置约束：此文件放在 internal/saas/ga/ 包下，而非策略包内，原因是避免策略包→ga包→策略包的导入循环。
```

### 5B. GA 主引擎

#### Prompt
```
请阅读 docs/进化计算引擎.md，然后实现 internal/saas/ga/engine.go。

EvolutionEngine 结构体字段包含：
- evolvable EvolvableStrategy（策略适配器接口，引擎唯一的策略通信渠道）
- 对 genomeStore 和 db 的依赖
- 以下可配置超参（括号内为默认值）：PopSize(300) / MaxGenerations(25) / EliteCount(8) / MutationProbability(0.15) / MutationScale(1.0) / MutationProbabilityMax(0.55) / MutationScaleMax(3.0) / MutationRampFactor(1.25) / EarlyStopPatience(5) / EarlyStopMinDelta(0.001) / TournamentSize(3)

EpochConfig 结构体包含：PopSize / MaxGenerations / LotStepSize / LotMinQty / OnProgress 进度回调函数 / SpawnPointOverride（非 nil 时覆盖冠军或默认的出生点）

RunEpoch 函数完整逻辑（严格按文档 docs/进化计算引擎.md 第 2 章实现）：

步骤一：构建 EvaluablePlan。从数据库拉取历史 K 线，调用 BuildCrucibleWindows 构建四个坩埚窗口；对每个窗口调用 SimulateGhostDCA 预计算 DCA 基线；封装为 EvaluablePlan（Epoch 内不可变）。

步骤二：种群初始化。从数据库加载精英基因列表（通过 DecodeElite 解码）。index 0 始终为当前种子冠军原样。其余个体按比例分配：约 10% 为精英原样、约 40% 为精英加强化变异（固定 prob=0.15 scale=1.5）、约 50% 为完全随机（通过 Sample 生成）。无精英时 index 0 为默认种子，其余全部随机。

步骤三：并发评估初始种群（见 evaluatePopulation 函数说明）。

步骤四：主进化循环（遍历 MaxGenerations 代）：
- 按适应度降序排序种群
- 收敛检测：当代最优 - 历史最优 < EarlyStopMinDelta，则 patienceCount++；否则更新历史最优，patienceCount=0
- 触发变异斜坡：patienceCount >= EarlyStopPatience 时，mutProb *= MutationRampFactor（上限 MutationProbabilityMax），mutScale *= MutationRampFactor（上限 MutationScaleMax）
- Early Stop：mutProb 和 mutScale 均已触及上限且仍无改善时退出循环
- 调用进度回调 OnProgress（当前代、最佳适应度、当前变异参数）
- 产生下一代：精英保留（Top EliteCount 直接入下一代），其余通过 tournamentSelect + Crossover + Mutate 产生

步骤五：取最优个体，调用 EncodeResult 序列化，写入数据库（Role = "challenger"），返回 EpochResult。

evaluatePopulation 函数：并发评估整个种群，带指纹缓存去重。
- Workers = min(runtime.NumCPU(), len(population))
- 用带缓冲 channel 作为任务队列，每个 worker 从队列取任务
- 对每个基因先计算 Fingerprint，命中缓存则直接复用，否则调用 evolvable.Evaluate
- 用 sync.Map 存储 fingerprint→fitness 的缓存
- 所有 worker 完成后返回 []float64 分数数组

tournamentSelect 函数：从种群中随机抽取 TournamentSize 个不同个体，返回适应度最高者。
```

### 5C. 坩埚窗口构建

#### Prompt
```
请实现 internal/quant/crucible.go，定义多时段坩埚切片的数据结构和构建逻辑。

CrucibleWindow 结构体：Label（"6m"/"2y"/"5y"/"full"）/ Weight（适应度权重）/ Bars（含 warmup 前缀的 K 线切片）/ EvalStartMs（评估区间起点时间戳，warmup 之后的第一根 bar）

CrucibleResult 结构体：Window 标签 / Score 分数 / ROI / MaxDD / Alpha（相对 Ghost DCA 的超额收益）

BuildCrucibleWindows 函数：输入全量历史 bars（按时间升序）和 warmupDays（指标预热天数，建议 1200 天），输出四个窗口切片，按 bar 数量升序排列（短→长，匹配级联短路顺序）。

四个窗口的构建规则：
- "6m"：评估区间从最新 bar 往前 183 天，评估区间之前再加 warmupDays 天的 warmup 前缀，Weight = 0.10
- "2y"：评估区间 730 天，同上加 warmup，Weight = 0.20
- "5y"：评估区间 1825 天，同上加 warmup，Weight = 0.30
- "full"：评估区间使用数据库中最早可用 bar 至最新 bar，不设人工天数上限，Weight = 0.40

严禁未来数据泄露：每个窗口的评估区间必须从 EvalStartMs 开始，warmup 数据必须在 EvalStartMs 之前，任何情况下不得让评估区间的计算看到 EvalStartMs 之后的数据。
```

### 5D. 进化任务服务与 HTTP Handler

#### Prompt
```
请阅读 docs/进化计算引擎.md 第 7 章，然后实现进化任务的服务层和 HTTP Handler。

internal/saas/epoch/service.go：

EpochService 结构体持有 db / EvolutionEngine / logger，以及一个互斥锁保护的 currentTask 指针（同时只允许运行一个进化任务）。

CreateAndRunTask 函数：
- 检查是否已有任务在运行，是则返回错误
- 解析 CreateTaskRequest（pop_size / max_generations / spawn_mode / spawn_point）
- 在 DB 创建 EvolutionTask 记录（Status="running"）
- 异步启动 runEpoch goroutine（不阻塞 HTTP 响应）
- 返回任务记录

spawn_mode 处理逻辑：
- "inherit"：从 DB 加载当前 champion 的 SpawnPoint，没有则用系统默认值
- "random_once"：调用 RandomSpawnPoint() 采样一次并冻结（整个 Epoch 共享）
- "manual"：使用请求体中的 spawn_point 字段

internal/saas/api/handler_evolution.go：实现以下三个 Handler：
- POST /api/v1/evolution/tasks：创建并启动进化任务，仅 lab/dev 模式可用
- GET /api/v1/evolution/tasks：返回当前任务状态 + 历次 challenger 列表（含 ScoreTotal / MaxDrawdown / 各窗口分数）
- POST /api/v1/evolution/tasks/:taskID/promote：人工审批晋升，在 DB 事务中执行：当前 champion→retired，challenger→champion；然后删除 Redis champion 缓存 key
- GET /api/v1/genome/champion：返回当前冠军基因包，优先从 Redis 缓存读取，cache miss 则从 DB 加载并写入缓存
```

---

## Phase 6 — 实例生命周期 + Cron Tick 驱动

### 目标
实现策略实例的创建/启停/删除，以及 cron 调度器驱动的 `Step()` 执行循环。

### Prompt
```
请阅读 docs/系统总体拓扑结构.md 第 6 章，然后实现实例生命周期管理模块。

internal/saas/instance/manager.go：

实例状态机：STOPPED → RUNNING（Start），RUNNING → STOPPED（Stop），任何状态 → DELETED（Delete），RUNNING → ERROR（异常）

Tick 函数（由 cron 每分钟扫描 RUNNING 实例时调用）：
步骤一：幂等桶去重检查。从行情数据源公开 API 拉取最新 K 线（按实例的 t_micro 聚合周期），获取最新已完成 bar 的时间戳，如果该时间戳 <= PortfolioState.LastProcessedBarTime，则跳过本次 tick（同一聚合桶已处理）。
步骤二：从 DB 读取实例的 PortfolioState（账户快照）和 RuntimeState（策略内部状态）。
步骤三：从 DB 或 Redis 加载当前冠军参数包，解析为策略 Params。
步骤四：ACL 外圈处理——将 []Bar 提取为 closes []float64 和 timestamps []int64。
步骤五：构建 StrategyInput（含 Portfolio 快照 + closes + timestamps + 文档要求的其他参数）。
步骤六：调用 [策略名].Step() 获取 StrategyOutput（这是唯一调用 Step() 的地方，与回测完全相同的函数）。
步骤七：持久化 RuntimeState。
步骤八：处理底仓释放意图——只更新 SaaS 侧账本中的 lot 分类，不向 Agent 下发任何指令，必须写 AuditLog。
步骤九：将 StrategyOutput 中的宏观/微观订单意图翻译为 TradeCommand，格式为 client_order_id = inst{id}-{engine}-{ts}，在 DB 写入 pending SpotExecution 记录，通过 WebSocket Hub 下发给对应 Agent。
步骤十：更新 LastProcessedBarTime。

如果 Agent 当前未连接，步骤九记录警告日志并跳过下发，等待下次 tick 重试。

internal/saas/cron/scheduler.go：
启动 cron 基础扫描（每分钟一次），遍历所有 RUNNING 实例，为每个实例并发启动 Tick goroutine。
```

---

## Phase 7 — LocalAgent（本地执行端）

### 目标
实现极简的本地执行二进制：从 SaaS 接收 TradeCommand，调用券商 API，上报 DeltaReport。

### Prompt
```
请阅读 docs/系统总体拓扑结构.md 第 5 章，然后实现 internal/agent/ 目录下的 LocalAgent。

铁律：Agent 不含任何策略代码；API Key 只存在于 config.agent.yaml；此文件必须在 .gitignore 中。

internal/agent/config/config.go：
AgentConfig 结构体包含 SaaSURL / Email / Password / Broker 四个字段。
Broker 子结构包含 Name / APIKey / SecretKey（及券商若要求的附加鉴权字段，如交易密码）/ Simulated（是否连接模拟盘）。
创建 config.agent.yaml 模板（所有密钥字段标注为"填写你的真实值"），并在 .gitignore 中排除此文件。

internal/agent/broker/client.go：
封装券商 REST API（A 股或黄金现货/ETF 通道，示例：华泰 OpenAPI）下单接口，只需实现两个方法：
- PlaceOrder(cmd TradeCommand) (Execution, error)：买入时以金额字段（如 OrderAmount / 委托金额，字段名随券商 API 而定）指定 CNY 金额的市价买单，卖出时以 Quantity 指定标的份额数量的市价卖单
- GetBalances() ([]Balance, error)：获取账户中所有资产的可用和冻结余额

internal/agent/ws/client.go：
AgentClient 主循环，含完整的自动重连逻辑：
- 重连策略：初始等待 1 秒，每次翻倍，最大等待 5 分钟
- 每次连接建立后的流程：
  1. 调用 SaaS REST API /api/v1/auth/login 获取 JWT
  2. 建立到 SaaS /ws/agent 的 WebSocket 连接
  3. 立即发送 auth 消息（携带 JWT）
  4. 等待 auth_result 确认
  5. 立即发送初始 DeltaReport（当前券商账户余额快照，client_order_id 为空）
  6. 进入消息循环
- 消息循环处理：
  收到 command 消息时：立即发 command_ack（不等执行完成），然后 goroutine 异步执行下单，完成后发 delta_report（含 client_order_id + 成交明细 + 当前余额）
  收到 heartbeat_ack 时：忽略
  每 30 秒发送一次 heartbeat 消息
```

---

## Phase 8 — WebSocket Hub（SaaS 侧）

### Prompt
```
请阅读 docs/系统总体拓扑结构.md 第 5 章，然后实现 internal/saas/ws/ 目录下的 WebSocket Hub。

设计原则：云端只信上报，端侧无脑执行。

ws/hub.go：连接管理中心
- 用 sync.Map 维护 userID → AgentConn 的映射（每个用户最多一个 Agent 连接）
- SendToAgent(userID, cmd TradeCommand) error：向指定用户的 Agent 发送指令，如 Agent 未连接返回错误
- HandleConnection(c *gin.Context)：处理新的 WebSocket 连接（路由：GET /ws/agent），流程如下：
  1. HTTP 升级为 WebSocket 连接
  2. 设置 10 秒超时等待第一条消息
  3. 验证第一条消息必须是 auth 类型，解析并验证 JWT
  4. 验证通过后注册连接，发送 auth_result 成功
  5. 进入消息循环：heartbeat → 回 heartbeat_ack；delta_report → 调用 processDeltaReport
  6. 连接断开时从 Map 中移除

ws/portfolio.go：DeltaReport 处理逻辑
processDeltaReport 函数流程：
1. 根据 client_order_id 找到对应的 pending SpotExecution 记录，更新为 filled 状态
2. 根据 SpotExecution 的 LotType 字段更新 PortfolioState：DEAD_STACK 成交更新 DeadHold，FLOATING 成交更新 FloatHold
3. 写入 TradeRecord
4. 用 DeltaReport.Balances 更新 PortfolioState 中的余额快照（真实数据来自券商账户）
5. 写审计日志
6. 发送 report_ack
注意：DeltaReport 中 client_order_id 为空时（Agent 重连初始快照），只更新余额快照，不更新 lot 记录。
```

---

## Phase 9 — REST API 路由

### Prompt
```
请实现 internal/saas/api/routes.go 以及各 Handler 文件。

路由结构如下，中间件说明：所有 /api/v1/ 下的非 auth 路由需要 JWT 鉴权；lab/evolution 相关路由额外需要 app_role = lab 或 dev 的检查。

公开路由（无需 JWT）：
POST /api/v1/auth/register
POST /api/v1/auth/login

用户路由（需要 JWT）：
GET  /api/v1/strategies             列出所有策略模板
GET  /api/v1/strategies/:id         获取策略模板详情
GET  /api/v1/instances              列出用户的策略实例
POST /api/v1/instances              创建实例（需检查订阅配额）
POST /api/v1/instances/:id/start    启动实例
POST /api/v1/instances/:id/stop     停止实例
DELETE /api/v1/instances/:id        删除实例
GET  /api/v1/instances/:id/lots     获取实例仓位详情
GET  /api/v1/instances/:id/trades   获取实例成交历史
GET  /api/v1/dashboard              获取账户总览数据
GET  /api/v1/agents/status          当前 Agent 连接状态

Lab 专属路由（需要 JWT + lab/dev role）：
POST /api/v1/evolution/tasks        创建并启动进化任务
GET  /api/v1/evolution/tasks        查询任务状态和历次 challenger 列表
POST /api/v1/evolution/tasks/:id/promote  人工审批晋升 challenger
POST /api/v1/backtests              触发单次回测（指定参数包）
GET  /api/v1/backtests/:id          获取回测结果
GET  /api/v1/genome/champion        获取当前冠军基因包
GET  /api/v1/genome/challengers     列出历次 challenger

WebSocket 路由：
GET  /ws/agent    Agent 长连接（走 Hub.HandleConnection）
```

---

## Phase 10 — 系统入口（cmd 层）

### Prompt
```
请实现 cmd/saas/main.go 和 cmd/agent/main.go。

cmd/saas/main.go 的启动顺序：
1. 读取 config.yaml，初始化 zap logger
2. 建立 DB 连接，执行 AutoMigrate（所有模型一次性完成）
3. 建立 Redis 连接
4. 初始化 WebSocket Hub
5. 初始化实例管理器，从 DB 恢复所有 RUNNING 状态的实例到内存
6. 初始化 GA 进化引擎（lab/dev 模式时实际可用，saas 模式时路由层拦截）
7. 启动 Cron 调度器
8. 启动 Gin HTTP 服务器
9. 监听 SIGTERM/SIGINT，收到信号后执行优雅停机：
   - 停止接受新请求和新 cron 任务
   - 等待正在运行的 tick 完成（超时 30s）
   - 持久化所有活跃实例的 RuntimeState 快照
   - 关闭所有 WebSocket 连接
   - 关闭 DB 和 Redis 连接

cmd/agent/main.go 的启动顺序：
1. 读取 config.agent.yaml，初始化 logger
2. 初始化券商交易客户端（用配置文件中的 API Key）
3. 初始化 AgentClient，启动主连接循环（含自动重连）
4. 监听 SIGTERM/SIGINT 优雅退出
```

---

## Phase 11 — 测试与验证

### Prompt
```
请为以下关键模块编写单元测试，测试目标是验证行为正确性，不是追求覆盖率数字。

一、Sigmoid 动态天平测试（internal/quant/micro_engine_test.go）
验证以下性质（每条写一个独立测试用例）：
- 当 Signal > 0 时，TargetWeight 必须 < 0.5
- 当 Signal < 0 时，TargetWeight 必须 > 0.5
- 当 CurrentWeight = 0.5 且 Gamma > 0 时，TargetWeight = 0.5（仓位恰好在中性点时偏置为零）
- 相同输入两次调用，结果完全一致（纯函数确定性）
- IsQuiet=true 时，|TheoreticalCNY| < 100 的情况 OrderCNY 必须为 0
- IsQuiet=false 且满足楔形突破条件时，|TheoreticalCNY| < 100 的情况 OrderCNY 必须等于 ±100

二、回测确定性测试（internal/adapters/backtest/adapter_test.go）
用真实历史数据（或生成的合成数据），相同参数跑两次回测，断言所有输出字段完全一致。

三、GA 引擎行为测试（internal/saas/ga/engine_test.go）
- 精英保留验证：一代进化后，上一代 Top N 个体必须出现在新种群中
- 变异斜坡验证：模拟 EarlyStopPatience 代无改善后，mutProb 应恰好等于初始值 × MutationRampFactor
- Fatal 个体验证：score=-99999 的个体经过 1000 次锦标赛选择，被选中次数应极少（< 5%）

四、WebSocket 协议测试（internal/saas/ws/hub_test.go）
- 未认证连接 10 秒后应自动断开
- 发送有效 DeltaReport 后，对应实例的 PortfolioState 应正确更新

最后运行完整测试套件并修复所有失败：
go test ./... -race -timeout 300s
```

---

## Phase 12 — Web 前端

### 设计基调

**视觉主题：宇宙暗夜终端（Cosmic Dark Terminal）**

写在 `web-frontend/` 目录。整体风格以沉浸式深空背景为底，融合量化终端的信息密度与现代 SaaS 的设计质感。核心视觉语言如下：

**背景与氛围**

页面底层颜色为接近纯黑的 `#020617`（deep navy-black），三个半透明光晕叠加在背景上营造空间感：珊瑚色 `#ff8c6b`（左上区域）、天空蓝 `#0ea5e9`（右下区域）、青绿色 `#2dd4bf`（中央区域），均施加 100–140px 的 blur 并以 `mix-blend-screen` 模式叠加。背景全局覆盖噪点纹理（fractal noise SVG，`mix-blend-color-dodge`，约 15% 透明度），模拟高端量化工具的颗粒感。两个几何装饰形状浮动在背景上层：左侧五边形（青绿色边框 + 渐变填充）、右侧平行四边形（珊瑚色边框 + 渐变填充），配合慢速浮动动画（motion/react）。支持 `prefers-reduced-motion`：若用户偏好减少动效，降级为静态背景（移除所有 motion 动画，保留视觉效果）。

**主色板（Tailwind 自定义 token）**

- 强调色（Accent）：`#2dd4bf`（青绿 / teal）——激活态导航、边框高亮、进度条
- 暖色（Warm）：`#ff8c6b`（珊瑚橙）——品牌 icon、警告/注意
- 信息色（Info）：`#0ea5e9`（天空蓝）——图表辅助线、标签
- 成功色：`#34d399`（绿）——盈利、健康状态
- 危险色：`#f87171`（红）——亏损、错误状态
- 警告色：`#fbbf24`（黄）——警告状态
- 文本主色：`#e2e8f0`（slate-200）
- 文本次色：`#94a3b8`（slate-400）
- 文本弱色：`#64748b`（slate-500）

**字体**

- 界面字体：Inter（`font-sans`）
- 数字 / 代码字体：JetBrains Mono（`font-mono`）——所有金额、比例、时间戳均使用

**卡片与组件风格**

玻璃拟态（glassmorphism）：`backdrop-blur`、`border border-white/[0.04]`、`bg-white/[0.02]` 或 `bg-slate-900/40`。激活态导航项：青绿色边框 `border-[#2dd4bf]/10` + 背景 `bg-[#2dd4bf]/[0.06]` + 文字 `text-[#2dd4bf]`。细滚动条：宽度 6px，颜色 `#334155`，轨道透明。

**动效原则**

Transition 使用 `duration-150`，平滑但不拖沓。数字更新无翻转动画（保持可读性优先）。页面内容区滚动流畅，侧边栏固定不滚动。

---

**UI 文案原则（面向用户的语言）**

所有技术术语必须翻译为用户可理解的商业化描述，用户界面中不得出现：
- 希腊字母裸露（β、γ 等，如必须展示须加括号说明）
- 内部状态机术语（如 `DEAD_STACK`、`S3_Panic`）
- 无上下文的数学指标名（如 `TheoreticalCNY`、`VolatilityRatio`）

参考替换规则：

| 内部术语 | 用户界面展示 |
|---|---|
| DeadStack / DeadHold | 长期持仓 |
| FloatStack / FloatHold | 活跃仓位 |
| ColdSealed | 封存资产 |
| TotalEquity | 总资产 |
| SpendableCNY | 可用资金 |
| Step() 触发 | 策略决策 |
| RUNNING | 运行中 |
| STOPPED | 已暂停 |
| ERROR | 异常 |
| challenger | 候选参数 |
| champion | 当前最优参数 |
| GA 进化任务 | 参数优化 |
| MarketState | 市场环境 |

---

**技术选型**

- 框架：React 18 + TypeScript + Vite
- 样式：Tailwind CSS v4（`@import 'tailwindcss'`，`@theme` 定义自定义 token）
- 组件库：shadcn/ui（暗色主题）
- 动效：motion/react（`framer-motion` 的 ESM 版）
- 图表：Recharts（NAV 曲线、持仓比例等）
- 状态管理：Zustand（`authStore`、`systemStatusStore`）
- 服务端状态 / 请求缓存：TanStack Query（`@tanstack/react-query`）
- HTTP 封装：原生 fetch，统一 error 类（`ApiRequestError`）
- 路由：React Router v6（`BrowserRouter` + `<Routes>`）
- 国际化：自实现轻量 i18n（`useI18n` hook + `locale` store + JSON 语言包）
- 图标：lucide-react

---

### 页面地图与跳转关系

```
/login                    登录页（AuthScaffold 布局）
/register                 注册页（AuthScaffold 布局）

/ (AppShell 布局：左侧边栏 + 顶部状态栏 + 主内容区)
├── /                     Dashboard 总览（实例卡片 + NAV 曲线 + 策略旅程卡片）
│   └── ?instance=:id     URL 参数选中指定实例，自动激活对应卡片
│
├── /templates            策略模板目录（Template 卡片列表，品牌色区分策略类型）
│
├── /instances            实例列表（所有实例状态概览，支持删除）
│   └── /instances/new    创建实例（策略模板选择 + 资金配额填写）
│
├── /evolution            进化实验室（仅 strategies feature 开启时可见）
│   ├── Tab: optimize     进化任务触发 + 进度监控 + 候选参数审批
│   └── Tab: library      基因库（历次 challenger/champion 历史记录）
│
├── /agents               Agent 管理（连接状态 + API Key 配置入口）
│
├── /backtesting          回测触发与结果展示（仅 backtesting feature 开启）
│
└── /settings             账户设置（底部导航项）
```

**AppShell 布局结构**

左侧边栏（Sidebar）：宽度在移动端收窄为纯图标模式（w-16），桌面端展开为图标 + 文字（w-64）。品牌 icon 区（珊瑚色 `Activity` 图标）固定在顶部。导航项从 `navItems` 配置中读取，通过 `hasFeature()` 函数控制是否渲染（feature flag 按 app_role 下发）。`Settings` 固定在侧边栏底部（`placement: 'footer'`）。

顶部状态栏（Topbar）：展示引擎运行状态（running / paused / halted 三态）、API 连接状态（Agent 是否在线）、用户邮箱 + 登出入口。每 30 秒轮询 `GET /api/v1/system/status`，若需要人工 Reconcile 则弹出 ReconciliationModal。

主内容区：`min-h-0 flex-1 overflow-y-auto p-4 lg:p-6`，内容最大宽度 1800px 居中，使用 Bento Grid（`.qs-bento-grid`，4 列响应式网格）组织卡片。

---

### 12A. 项目脚手架与主题系统

#### Prompt
```
请在项目根目录的 web-frontend/ 下初始化前端项目，完成视觉主题配置，并搭建 AppShell 布局骨架。

技术栈要求：
React 18 + TypeScript + Vite，Tailwind CSS v4，shadcn/ui（暗色），motion/react，TanStack Query，Zustand，React Router v6，lucide-react，JetBrains Mono + Inter 字体（Google Fonts）。

视觉主题核心要求：

1. 全局背景色：#020617（纯黑深蓝）
2. 在 @theme 块中定义以下 CSS 变量 token（供 Tailwind 使用）：
   - qs-bg: #0f1115
   - qs-surface: rgba(255,255,255,0.04)
   - qs-accent: #2dd4bf（青绿，导航激活、进度条、高亮边框）
   - qs-danger: #f87171
   - qs-warn: #fbbf24
   - qs-safe: #34d399
   - font-sans: 'Inter'
   - font-mono: 'JetBrains Mono'

3. 全局动态背景组件（AppBackground）：
   - 三个大光晕球（Framer Motion 慢速循环动画）：
     * 左上：#ff8c6b 珊瑚色，blur-[120px]，mix-blend-screen
     * 右下：#0ea5e9 天空蓝，blur-[140px]，mix-blend-screen
     * 中央：#2dd4bf 青绿色，blur-[100px]，mix-blend-screen
   - 动态浮动粒子（20 个白色小点，从底部飘向顶部，循环动画）
   - fractal noise 噪点纹理叠加（opacity 约 0.15，mix-blend-color-dodge）
   - 两个几何装饰形状（五边形：青绿色；平行四边形：珊瑚色），慢速浮动
   - 支持 prefers-reduced-motion：检测到时改用 StaticBackdrop（仅静止光晕，无粒子动画）

4. 细滚动条样式（.custom-scrollbar）：宽度 6px，颜色 #334155，轨道透明。

5. Bento Grid 系统（.qs-bento-grid）：
   - 移动端：1 列
   - 平板（md）：2 列
   - 桌面（lg）：4 列，支持子元素通过 col-span-{1..4} 控制跨列

AppShell 布局骨架要求：

左侧 Sidebar：
- 固定高度 h-screen，左边框 border-r-2 border-[#0a0f1c]，背景 bg-[#020617]/40，backdrop-blur-xl
- 品牌区：高 h-16，内含 Activity 图标（#ff8c6b 珊瑚色，带 glow shadow），右侧品牌名称文字（桌面端显示，移动端隐藏）
- 导航区：从 navItems 配置数组渲染 NavLink，通过 hasFeature() 过滤；激活样式：青绿色文字 + 边框 + 内发光背景；非激活：slate-500 文字，hover 略亮
- Settings 固定在底部 footer 区域

顶部 Topbar：
- 高 h-16，右侧展示三个状态指示器：API Key 是否配置、Agent 是否在线、引擎运行状态（running/paused/halted 三态，颜色分别为绿/黄/红）
- 右侧展示用户邮箱 + 下拉登出按钮

AuthProvider（src/app/AuthProvider.tsx）：
- 应用启动时从本地存储恢复 JWT，调用 GET /api/v1/auth/me 验证有效性
- 提供 user、loading、login、logout 方法

AppRouter（src/app/router.tsx）：
- /login 和 /register 使用 AuthScaffold（居中布局，背景复用 AppBackground）
- / 根路由使用 AppShell，内嵌 AuthGate（未登录重定向 /login）
- 各功能路由通过 hasFeature() 决定是否渲染，无权限时 Navigate 回首页
```

---

### 12B. 认证页（Login + Register）

#### Prompt
```
请实现 src/features/auth/LoginPage.tsx 和 RegisterPage.tsx，布局使用 AuthScaffold（居中，背景为 AppBackground，支持暗色光晕效果）。

视觉要求：
- 卡片宽度 400px，背景半透明（玻璃拟态：border border-white/10, backdrop-blur-xl, bg-slate-900/60）
- 顶部：Activity 图标（珊瑚色 #ff8c6b）+ 系统名称（大号字，slate-200）+ 简短 slogan
- 表单字段：邮箱 + 密码（登录），邮箱 + 密码 + 确认密码（注册）
- 输入框：暗色背景 bg-slate-900/80，border-slate-700，focus 时边框变为青绿色 #2dd4bf
- 提交按钮：全宽，青绿色背景，加载中显示 Loader2 旋转图标并禁用
- 错误提示：红色小字，显示在按钮下方
- 底部切换链接："没有账号？注册" / "已有账号？登录"

交互逻辑：
- 登录/注册成功后通过 AuthProvider 的 login() 方法存储 JWT，跳转至 /
- 表单提交期间禁用所有输入和按钮
- 字母间距使用 tracking-wider，按钮文字 uppercase

对接接口：POST /api/v1/auth/login 和 POST /api/v1/auth/register
```

---

### 12C. Dashboard 总览页

#### Prompt
```
请实现 src/features/dashboard/DashboardPage.tsx，这是用户登录后的主页。

页面整体采用 Bento Grid（.qs-bento-grid）布局，卡片使用玻璃拟态风格（半透明边框，backdrop-blur）。

左侧实例选择区（桌面端约占 1/4 宽度）：
- 列出所有策略实例，以卡片列表形式展示
- 每张实例卡片：实例名称 + 标的代码 + 状态徽章（运行中/已暂停/异常，颜色对应青绿/灰/红）
- 点击卡片选中该实例，激活态用青绿色左边框高亮
- 卡片底部：启动/暂停按钮（运行中显示暂停，已暂停显示启动），点击直接调用 PATCH /api/v1/instances/:id
- 页面顶部有"前往配置"入口（Settings 图标按钮），跳转至 ConfigFormSheet（侧滑面板）
- 底部有"新建实例"按钮，跳转 /instances/new
- URL 参数 ?instance=:id 支持直接选中指定实例（页面加载时自动激活）

右侧主展示区（约占 3/4 宽度，纵向排列）：

上方：策略概况卡片（StrategyOverviewCard）
- 展示已选实例的总资产、资产分布（长期持仓 / 活跃仓位 / 可用资金 / 封存资产）
- 显示实例当前状态、最后决策时间
- 数字使用 font-mono

中部：PnL 净值曲线（PnLChart 组件，Recharts AreaChart）
- X 轴：时间，Y 轴：总资产（CNY）
- 折线色为青绿 #2dd4bf，面积半透明填充
- 支持时间范围切换（7天/30天/90天），当前选中项用青绿色高亮
- 数据来源：GET /api/v1/dashboard/equity-snapshots?instance_id=:id
- 加载中显示 Skeleton（PnLChartSkeleton）

下方：策略旅程卡片（StrategyJourneyCard）
- 展示当前实例的策略运行关键里程碑（首次运行时间、累计决策次数、本月成交次数等）
- 用户友好语言，不暴露内部字段名

数据轮询：实例列表每 60 秒刷新，equity 曲线每 60 秒刷新（TanStack Query refetchInterval）。
```

---

### 12D. 策略模板页 + 实例管理页

#### Prompt
```
请实现以下三个页面：

一、src/features/strategies/TemplatesPage.tsx（/templates）
展示所有可用的策略模板，以卡片网格排列。每张模板卡片包含：
- 策略名称（用户可理解的商业化名称，如"动态均衡策略"，而非 lunar-equity 等内部代号）
- 支持的标的代码与券商通道
- 简短策略描述（不超过 2 行，聚焦用户价值，不描述内部算法）
- 策略色彩标识（每个策略有专属的品牌色，用于卡片左边框/顶部装饰条）
- 是否支持进化优化的标签
- "创建实例"按钮，点击跳转 /instances/new?template=:id

模板数据来自本地 strategyCatalog.ts 配置文件（静态配置，无需从后端获取），该文件定义每个策略的 UI 展示属性：名称、描述、色彩、支持的 feature。

二、src/features/strategies/InstanceListPage.tsx（/instances）
展示当前用户的所有实例，以列表形式排列（非网格）。每行包含：
- 实例名称 + 策略类型 + 标的代码
- 状态徽章
- 总资产（font-mono）
- 创建时间（相对时间，如"3天前"）
- 操作列：跳转 Dashboard（?instance=:id）、跳转进化页（/evolution?instance=:id）、删除按钮（需二次确认）
删除时显示 Trash2 图标，点击后出现内联确认按钮（避免 modal 打断流程）。
顶部有"创建新实例"按钮跳转 /instances/new。

三、src/features/strategies/InstanceCreatePage.tsx（/instances/new）
分步表单（两步）：
第一步：选择策略模板（展示与 TemplatesPage 相似的卡片，选中后高亮，点击下一步）
第二步：填写实例配置：
  - 实例名称（自定义标识）
  - 初始资金配额（CNY）
  - 月度注资金额（CNY，可选）
  - 封存资产量（可选，"永不释放"的底仓）
  - 风险偏好（最大可用回撤，滑块选择）
提交后调用 POST /api/v1/instances，成功后跳转 /（Dashboard），URL 带 ?instance=:id 自动聚焦新实例，并通过 React Router location.state 展示成功 notice。
```

---

### 12E. 进化实验室页

#### Prompt
```
请实现 src/features/strategies/EvolutionPage.tsx（/evolution）。

页面顶部：实例选择器（下拉或卡片列表，仅显示支持进化的实例），URL 参数 ?instance=:id 支持直接选中。

页面主体分两个 Tab（optimize / library）：

Tab: optimize（参数优化）
分三个区域从上到下排列：

1. 进化控制面板（EvolutionPanel 组件）
   - 如无运行中任务：展示"启动新一轮优化"按钮，点击后展开参数配置区
     * 种群大小（10-500，默认300）
     * 最大代数（5-50，默认25）
     * 参数继承方式（单选：继承当前最优 / 随机探索 / 手动指定）
     * 手动指定时：JSON textarea 输入区，带格式提示
     * 提交按钮调用 POST /api/v1/evolution/tasks
   - 如有运行中任务：展示任务进度卡
     * 当前代 / 最大代进度条（青绿色）
     * 当前最优评分（数字，font-mono）
     * 最大回撤（红色显示，font-mono）
     * 每 5 秒轮询 GET /api/v1/evolution/tasks 刷新
     * "终止任务"按钮（需确认）

2. 任务队列视图（TaskQueueView 组件）：历史任务列表，每项展示状态、评分、耗时。

3. 冠军基因展示：当前 champion 的核心指标（综合评分、各窗口分数、最大回撤），用青绿色边框卡片高亮，顶部标注"当前最优参数"。附"应用到实例"按钮，点击后跳转 Dashboard 对应实例。

Tab: library（基因库）
展示所有历史基因记录（GenomeLibrary 组件）：
- 卡片列表，按时间倒序
- 每张卡片：角色标签（候选参数/当前最优/历史归档）、产出时间、综合评分、各窗口分数（6m/2y/5y/全量）、最大回撤
- 当前 champion 用青绿色边框高亮
- challenger 卡片有"晋升为最优"按钮，点击后有二次确认弹窗（说明晋升影响），调用 POST /api/v1/evolution/tasks/:id/promote
- 可跳转回测页查看该基因的完整回测报告（/backtesting?genome=:id）

数据来源：GET /api/v1/evolution/tasks、GET /api/v1/evolution/genomes
```

---

### 12F. Agent 管理页 + 回测页

#### Prompt
```
一、src/features/agents/AgentsPage.tsx（/agents）

页面展示 LocalAgent 的连接状态与配置指引。

上方：连接状态卡片
- 大号状态指示器：在线（青绿色脉冲圆点 + "执行端已连接"）/ 离线（灰色 + "执行端未连接，交易将暂停"）
- 最后心跳时间（font-mono）
- Agent 版本号（如已连接）
- 数据来源：SystemStatusContext 中的 api_connected 字段，每 30 秒自动刷新

中间：Agent 配置说明区
以步骤卡片形式引导用户完成 LocalAgent 配置：
Step 1: 下载 LocalAgent 二进制（提供下载链接占位）
Step 2: 创建 config.agent.yaml（提供配置模板，高亮提示：API Key 仅存本地，永不上传）
Step 3: 运行 Agent，确认连接

下方：API Key 配置检查
展示当前 SaaS 侧检测到的 API 配置状态（api_configured 字段）。
注意：API Key 本身永不展示在前端，仅显示"已配置"/"未配置"。

二、src/features/backtesting/BacktestingPage.tsx（/backtesting）

顶部：实例选择器（下拉，仅显示有 champion 基因的实例）

触发区（表单卡片）：
- 参数来源（单选）：使用当前最优参数 / 使用指定候选参数（下拉选 challenger）/ 自定义 JSON
- 点击"开始回测"，调用 POST /api/v1/backtests，显示加载动画

结果展示区（完成后渲染）：
- 概要指标卡片（Stats Cards）：总收益率 / 相对 DCA 的 Alpha / 最大回撤 / 夏普比率（如有），使用 font-mono
- NAV 曲线（Recharts AreaChart，青绿色）
- Ghost DCA 基准线叠加（虚线，slate-500 色），直观对比策略 vs 被动 DCA 的差距
- 各时间段分解：6个月/2年/5年/全量 评分卡片

数据来源：POST /api/v1/backtests（触发），GET /api/v1/backtests/:id（轮询结果，每3秒一次直至完成）
```

---

### 12G. 通用组件库与服务层

#### Prompt
```
请实现以下通用组件和服务层，供各页面复用。

一、src/shared/ui/Card.tsx
基础玻璃拟态卡片组件：border border-white/[0.04]，bg-slate-900/20，backdrop-blur，rounded-xl。
接受 className 进行样式扩展。

二、src/shared/ui/StatusBadge.tsx（或 StatusPill）
状态徽章：接受 status（running/stopped/error/halted）映射为颜色（青绿/灰/红/红）和用户友好文案（运行中/已暂停/异常/已中断）。圆角 pill 样式，带实心小圆点。

三、src/shared/ui/skeletons/
常用 Skeleton 占位组件：PnLChartSkeleton（灰色矩形占位区）、CardSkeleton、TableSkeleton。
背景色使用 bg-slate-800/40，配合 animate-pulse。

四、src/shared/services/
HTTP 服务层，统一封装 fetch 请求：
- 所有请求自动附加 Authorization: Bearer {token}（从 authStore 读取）
- 响应非 2xx 时抛出 ApiRequestError（含 status 和 message）
- 401 时触发 authStore.logout() 并跳转 /login
- 按业务域分文件：instances.ts / dashboard.ts / evolution.ts / backtests.ts / system.ts

五、src/shared/config/features.ts
Feature flag 系统：
- AppFeature 类型：'dashboard' | 'strategies' | 'agents' | 'risk' | 'backtesting' | 'settings'
- hasFeature(feature: AppFeature): boolean，从初始化 API 响应或 app_role 判断

六、src/shared/config/navigation.ts
navItems 数组配置：每项包含 to（路由）、labelKey（i18n key）、icon（LucideIcon）、placement（main/footer）、feature（可选，用于 hasFeature 过滤）、end（用于 NavLink 精确匹配）

七、src/i18n/
轻量 i18n 实现：
- I18nProvider 包裹应用，提供 locale（zh/en）切换
- useI18n() hook 返回 t(key) 翻译函数和当前 locale
- 语言包 JSON 文件：zh.json 和 en.json，覆盖所有 nav、common、页面内文案
- 所有面向用户的字符串通过 t() 获取，不硬编码

八、src/stores/authStore.ts（Zustand）
- 存储 user（含 email、role）和 JWT token
- 应用启动时自动从 localStorage 恢复 token 并验证（GET /api/v1/auth/me）
- 提供 login(token, user) / logout() / loading 状态
```

---

### 验收要点

- 所有金额保留 2 位小数，标的份额（股/份/克）保留 6 位小数，均使用 font-mono
- 页面全局搜索：不得在用户可见文案中出现 `DeadHold`、`FloatHold`、`Step()`、`VolatilityRatio`、`TheoreticalCNY`、`lunar` 等内部术语
- 主视觉背景（三色光晕 + 噪点 + 几何装饰）必须在所有页面一致呈现
- Sidebar 激活态必须使用青绿色（`#2dd4bf`）高亮，非激活态使用 slate-500
- 支持 `prefers-reduced-motion`：减动效模式下背景光晕静止、无浮动粒子
- 桌面端（lg 1024px+）Sidebar 展开为图标+文字（w-64），移动端收窄为纯图标（w-16）
- Feature flag 保证无权限路由自动重定向至首页，不显示 404
- 前端构建命令：`cd web-frontend && npm run build`，产物输出至 `web-frontend/dist/`
- SaaS 后端新增静态文件服务，将 `dist/` 托管在 `/` 根路径，API 路由保持 `/api/v1/` 前缀

---

## Phase 13 — Docker 部署配置

### Prompt
```
请创建生产部署所需的全部配置文件。

saas.Dockerfile：多阶段构建，builder 阶段使用 golang:1.21 编译 cmd/saas/main.go，最终镜像基于 alpine，只复制编译好的二进制文件和 config.yaml。

agent.Dockerfile：同上，编译 cmd/agent/main.go，最终镜像只包含 agent 二进制。

docker-compose.yml：适用于本地开发和 lab 模式，包含三个服务：
- postgres（postgres:15，创建 quantsaas 数据库，持久化到 named volume）
- redis（redis:7-alpine）
- saas（从 saas.Dockerfile 构建，APP_ROLE 通过环境变量注入，默认 dev，depends_on postgres 和 redis，暴露 8080 端口）

说明：agent 二进制设计为用户在本地直接运行（而不是容器），因为它需要访问用户本地的 config.agent.yaml 密钥文件。

.gitignore 必须包含：
config.agent.yaml
*.env
*.exe（Windows 编译产物）
bin/（本地编译输出目录）
```

---

## 完整验收检查清单

构建完成后，逐项确认以下检查项：

**架构铁律（用 grep 验证）：**
```bash
# 确认策略包内无 isBacktest 分支
grep -r "isBacktest" internal/strategies/      # 期望：无结果

# 确认 SaaS 侧无 API Key 字段
grep -r "api_key\|secret_key\|trade_password" internal/saas/  # 期望：无结果

# 确认策略内核不依赖 Bar 结构体
grep -r "quant\.Bar" internal/strategies/      # 期望：无结果

# 确认策略包内无 I/O 调用
grep -r "http\.\|sql\.\|os\.Open\|time\.Now" internal/strategies/  # 期望：无结果
```

**功能验证：**
- `go build ./...` 无错误
- `go test ./... -race` 全部通过
- 相同参数回测两次，所有输出字段完全一致
- GA 在 test_mode（Pop=10, Gen=3）下能完整运行并写入 challenger 记录
- 同一 bar 时间戳，Cron Tick 对同一实例不会重复推进（幂等验证）

**安全检查：**
- `config.agent.yaml` 在 `.gitignore` 中
- `git status` 确认 `config.agent.yaml` 未被追踪
- 未鉴权的 WebSocket 连接在 10 秒内断开
- app_role=saas 时访问 lab 专属路由返回 403

---

## 关键参数参考表

| 模块 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| GA | PopSize | 300 | 种群大小 |
| GA | MaxGenerations | 25 | 最大代数 |
| GA | EliteCount | 8 | 精英保留数量 |
| GA | TournamentSize | 3 | 锦标赛参与人数 |
| GA | MutationProbability | 0.15 | 初始变异概率 |
| GA | MutationScale | 1.0 | 初始变异幅度 |
| GA | MutationProbabilityMax | 0.55 | 变异概率上限 |
| GA | MutationScaleMax | 3.0 | 变异幅度上限 |
| GA | MutationRampFactor | 1.25 | 斜坡放大倍率 |
| GA | EarlyStopPatience | 5 | 无改善代数触发斜坡 |
| GA | EarlyStopMinDelta | 0.001 | 改善阈值 |
| 适应度 | Fatal 回撤阈值 | 88% | 超过则硬否决 |
| 适应度 | DD 超额惩罚系数 | 1.5× | 超额回撤的罚分倍率 |
| 微观 | EMA/σ 窗长 | 21 根 | 不可进化的固定常量 |
| 微观 | VolRatio 长窗口 | 112 根 | 楔形过滤分母 |
| 微观 | VolRatio 短窗口 | 16 根 | 楔形过滤分子 |
| 微观 | 最小订单阈值 | 100 CNY | 粉尘拦截边界（对齐 A 股常见最小成交单位；黄金 ETF 可按券商规则配置） |
| 系统 | 心跳间隔 | 30 秒 | Agent → SaaS |
| 系统 | 重连初始等待 | 1 秒 | 指数退避起点 |
| 系统 | 重连最大等待 | 5 分钟 | 指数退避上限 |
| 系统 | Auth 超时 | 10 秒 | 未认证连接自动断开 |
| 账本 | micro_reserve_pct 默认 | 0.25 | 微观层资金保留比例 |

---

## ⚙️ [OPTIONAL] Phase 13 — AI 多维信号层（LLM 辅助开单）

> **此 Phase 为可选扩展，不影响主系统运行。**
> 在核心系统（Phase 0–12）稳定运行之后，再考虑接入。
> 核心铁律不变：AI 信号的生成必须发生在 `Step()` 调用之前的 cron 外圈，LLM 调用结果以快照形式注入 `StrategyInput`，`Step()` 内部保持纯函数。

---

### 设计思路

当前 Sigmoid 动态天平的 `Signal` 由纯技术指标驱动。本 Phase 引入一个独立的 **AI 信号层**，让 LLM 周期性地阅读三类外部信息，生成一个三维信号向量，作为独立的附加项叠加进 Sigmoid 的 Exponent，与技术信号共同决定目标仓位。

**三个维度的定义：**

| 维度 | 含义 | 取值范围 | 信号方向 |
|---|---|---|---|
| S_market（行情面） | AI 对近期价格结构、关键位、宏观趋势的判断 | [-1, 1] | 正值 = 倾向减仓，负值 = 倾向加仓 |
| S_news（消息面） | AI 对近期新闻标题/公告的利多/利空评估 | [-1, 1] | 同上 |
| S_sentiment（情绪面） | AI 对市场情绪指标（A 股情绪指数、北向资金、黄金 ETF 溢价等）的综合判断 | [-1, 1] | 同上 |

**与 Sigmoid 动态天平的联动方式：**

AI 三维信号向量加权合成为一个标量 `AISignal`，以独立项叠加进 Exponent：

```
AISignal = w1 × S_market + w2 × S_news + w3 × S_sentiment

Exponent = EffectiveBeta × TechSignal
         + AIBeta × AISignal
         + γ × InventoryBias
```

其中 `w1 / w2 / w3 / AIBeta` 均为可进化参数，加入染色体，由 GA 搜索各维度的实际贡献权重。当 GA 发现某个维度对收益无贡献时，对应权重会自然收敛到 0。

**更新频率与缓存策略：**
AI 信号每 4 小时更新一次，结果存入 Redis，TTL 为 4 小时。cron tick 读缓存，cache miss 时触发一次 LLM 调用并写入缓存。LLM 调用失败时降级为 `[0, 0, 0]` 中性向量，系统继续正常运行，不因 AI 服务不可用而中断交易。

---

### 数据源设计

| 维度 | 数据来源 | 说明 |
|---|---|---|
| 行情面 | 系统自身 K 线数据库 | 从已有 KLine 表提取近期 OHLCV 摘要 + 关键均线位置，无需外部 API |
| 消息面 | 财经新闻 API（如东方财富、财联社） | 提供近 24 小时 A 股/黄金相关新闻标题与来源，按标的过滤 |
| 情绪面 | 市场情绪 API（如中证市场情绪指数） | 提供当日市场情绪读数；A 股可叠加北向资金/融资融券，黄金可叠加 ETF 溢价与持仓数据 |

---

### Context（写给 AI 的背景）

这一层的核心难点不在技术实现，而在 **Prompt Engineering**：如何让 LLM 稳定地输出结构化的三维评分，而不是自由发挥。LLM 返回的内容必须经过严格的解析和范围夹紧，任何解析失败都应静默降级，不抛出异常影响主流程。

---

### Prompt

```
请阅读 docs/策略数学引擎.md，理解 Sigmoid 动态天平的 Exponent 结构，然后实现 AI 多维信号层。

架构约束（必须遵守）：
- AI 信号的 LLM 调用发生在 cron tick 中，Step() 调用之前
- Step() 是纯函数，不得在内部发起任何网络请求
- AISignalVector 作为字段加入 StrategyInput 结构体，以快照形式注入

请实现以下内容：

一、数据采集层 internal/saas/ai/collector.go
实现三个采集函数，各自独立，互不依赖：

MarketSummary：从 KLine 表读取指定标的近期 K 线（近 7 天日线 + 近 24 小时小时线），计算并格式化以下摘要字符串：当前价格、距近期高点/低点的百分比距离、多条 EMA 的相对位置（价格高于/低于均线多少百分比）。输出为一段结构化文字，供后续 LLM 调用使用。

NewsSummary：调用财经新闻 API（如东方财富资讯接口或财联社开放接口）获取指定标的近 24 小时新闻标题列表，拼接为 bullet list 格式字符串。API Key 存在 config.yaml 中（非 agent 配置）。获取失败时返回空字符串。

SentimentSummary：调用市场情绪 API（如中证市场情绪指数或同类免费接口）获取当日情绪读数（数值 + 文字描述），格式化为一行文字。获取失败时返回 "市场情绪: 不可用"。

二、LLM 评分服务 internal/saas/ai/scorer.go
实现 ScoreAISignal 函数，接收三个摘要字符串，调用 Claude API（claude-haiku-4-5-20251001 模型，低成本），返回 AISignalVector{SMarket, SNews, SSentiment float64}。

System prompt 要求 LLM 扮演"资深 A 股/黄金量化分析师"，根据输入信息对三个维度各自打分，遵循以下规则：
- 分数范围 [-1.0, 1.0]，正值表示当前信号倾向降低仓位（偏空），负值表示倾向提升仓位（偏多），0 表示中性
- 必须以纯 JSON 格式返回，格式为 {"s_market": 0.0, "s_news": 0.0, "s_sentiment": 0.0}，不得包含任何额外文字
- 当信息不足或不确定时，对应维度输出 0

响应解析：严格解析 JSON，若解析失败或字段缺失则返回 [0, 0, 0] 中性向量；将所有值 clamp 到 [-1.0, 1.0]。LLM 调用超时设为 15 秒。

三、缓存服务 internal/saas/ai/cache.go
实现 GetCachedSignal 和 SetCachedSignal 两个函数，使用 Redis 存储 AISignalVector，key 格式为 ai_signal:{symbol}，TTL 为 4 小时。序列化方式使用 JSON。

四、信号编排服务 internal/saas/ai/service.go
实现 FetchAISignal(ctx, symbol) AISignalVector 函数：
先查 Redis 缓存，命中则直接返回；
未命中则串行调用 MarketSummary + NewsSummary + SentimentSummary，拼接后调用 ScoreAISignal；
写入缓存后返回；
任何环节出错均静默降级，返回 [0, 0, 0]，写 warn 日志，不返回 error 给上层。

五、注入 StrategyInput
在 quant/data.go 的 StrategyInput 结构体中新增字段 AISignalVector（三个 float64 字段 SMarket/SNews/SSentiment）。在 cron tick 的 Tick 函数中，Step() 调用之前，调用 FetchAISignal 并将结果写入 StrategyInput。

六、扩展 Chromosome
在 Chromosome 结构体中新增四个可进化参数：
- W1（行情面权重）：边界 [-2, 2]，默认 0（初始中性，由 GA 自行发现贡献）
- W2（消息面权重）：边界 [-2, 2]，默认 0
- W3（情绪面权重）：边界 [-2, 2]，默认 0
- AIBeta（AI 信号整体激进系数）：边界 [0, 3]，默认 0

七、扩展微观引擎
在 MicroDecisionInput 结构体中新增 AISignalVector 和对应的权重参数字段。
在 ComputeMicroDecisionV4 函数中，在现有 Exponent 计算之后叠加 AI 信号项：
AISignal = W1 × SMarket + W2 × SNews + W3 × SSentiment
Exponent += AIBeta × AISignal
其他逻辑不变。

八、config.yaml 新增配置项
在 config.yaml 中新增 ai 配置块，包含 claude_api_key（从环境变量 ANTHROPIC_API_KEY 注入）和 finance_news_api_key（从环境变量 FINANCE_NEWS_API_KEY 注入）。两个 key 均不得硬编码，不得出现在任何代码文件中。
```

---

### 验收要点

- `grep -r "ANTHROPIC_API_KEY\|api_key" internal/` 中不应出现硬编码密钥
- LLM 服务宕机时，回测和实盘均应正常继续（AISignalVector = [0,0,0]）
- GA 回测中 AISignalVector 由