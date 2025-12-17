## 击剑视频智能分析平台 · 产品白皮书（v1.0）

### 1. 执行摘要
击剑视频智能分析平台是一套面向教育训练与竞技复盘的端到端数据化工具。系统以比赛/训练视频为输入，自动完成"目标检测 → 跟踪与分割 → 回合切分 → 指标计算 → 图表生成 → 报告导出"，输出结构化数据与可视化报告，帮助教练、运动员与教学单位以客观量化的方式进行技术评估、战术复盘与阶段性改进。

平台基于 Flask+SQLAlchemy 后端与 Celery+Redis 异步架构，CV/ML 采用 YOLO、Track-Anything、PyTorch、MMCV 等技术栈。结果以 JSON/CSV/PNG/视频片段的形式保存到 `results/{user_id}/{upload_id}` 分层目录，支持左右击剑手分侧分析、跨回合趋势对比与标签检索。

#### 1.1 立项背景与问题现状
- 教练复盘与教学耗时：传统"逐帧观看+手工记录"方式效率低、主观偏差大，难以跨赛季、跨队伍对比；
- 数据沉淀不足：训练与比赛视频虽丰富，但缺少标准化指标与结构化数据沉淀，难以支撑持续改进；
- 教学评价不客观：缺少量化口径（速度、节奏、伸臂、冲刺、进攻/防守比例等），难以对学员进行客观评估；
- 科研与普及脱节：运动科学研究成果难以落地到基层俱乐部和校园场景，需要"可用、可负担"的工具；
- 裁判争议复盘缺支撑：赛后缺乏统一的数据与可视化证据，技术代表和裁判复盘成本高。

#### 1.2 建设目的与总体目标
- 建设目的：以视频为核心载体，形成"采集—分析—评估—改进"的数据化闭环，提高训练与复盘效率，沉淀可复用的知识资产；
- 总体目标：
  - 形成面向校园与俱乐部的一体化击剑分析平台，低门槛部署，上手即用；
  - 建立标准化的技战术指标口径与图表表达，输出可解释的分析结果；
  - 支持左右击剑手分侧建模、跨回合/跨视频对比与标签化管理；
  - 支持本地/私有部署与权限隔离，满足隐私合规要求；
  - 提供扩展接口，便于后续算法升级与系统对接。

#### 1.3 建设原则与设计理念
- 实用优先：界面简洁、关键路径短，上传即分析，结果即复盘；
- 标准口径：各指标定义统一、参数可配置、版本可追溯；
- 模块解耦：路由、任务、算法、数据、可视化分层清晰，便于维护与替换；
- 开放扩展：算法与图表插件化，便于项目组或第三方快速扩展维度；
- 稳健可靠：异步队列、失败重试与日志追踪，支持长视频、大批量场景；
- 安全合规：最小权限、结果目录隔离、可选HTTPS/反代，敏感配置外置。

#### 1.4 预期效益与绩效指标
- 教学效益：每场/每节课复盘时间缩短≥50%，可视化报告覆盖≥80%训练场景；
- 运动员发展：阶段指标（速度、节奏、伸臂、冲刺成功等）均有量化曲线；
- 组织管理：建立"选手档案+视频+指标"的完整资产库；
- 科研支撑：输出可对照的结构化数据，支撑校内课题或论文复现；
- 绩效指标：回合切分准确率、任务成功率、任务耗时P95、报告使用率、用户留存等。

### 2. 用户与价值
- 教练/教师：
  - 基于雷达与趋势的"训练处方"，针对首步、暂停控制、伸臂等短板设定周目标；
  - 课堂范例库：沉淀典型错误/优秀动作，支撑示范与考试评判；
  - 客观评估：期初-期中-期末三段曲线，减少主观争议与沟通成本。
- 运动员：
  - 自主复盘：回合片段+标签清单，对照自评与教练评语；
  - 成长跟踪：速度/节奏/冲刺成功率等指标"本周 vs 上周、赛前 vs 赛后"对比；
  - 目标激励：围绕"暂停控制/首步滞后"等单项设置周挑战与打榜。
- 俱乐部/队伍与教学组织：
  - 训练质量看板：覆盖率、复盘完成率、整体指标提升度；
  - 招生与宣传：可发布的图表与成长故事；家校沟通透明；
  - 科研支撑：结构化数据便于课题与论文复现。
- 裁判/赛事技术：
  - 赛后复盘：关键回合证据（片段+曲线+标签），用于复审与裁判培训；
  - 规则教学：右路权/节奏范例库。

- 场景化流程（摘要）：课前设目标 → 课后上传自动分析 → 教练批注与打标签 → 生成训练处方 → 月度汇总与展示（详见"附加D"）。
- 量化目标与ROI（摘要）：复盘时长-50%~ -60%；覆盖率≥80%；暂停比率↓≥10%；首步滞后↓≥20%；赛后复盘出具≤24h。
- 采纳路线（摘要）：第1月口径与基线 → 第2月覆盖全员闭环 → 第3月班级/小队看板 → 第4月赛事/科研/宣传应用（详见"附加D"）。

核心收益：客观量化、提效降本、知识沉淀、闭环改进。（详见"附加D · 用户价值详解"）

### 2A. 需求分解与验收口径（扩展）
下列关键需求以"需求条目—验收口径—度量指标—数据来源—备注"方式列示，正式交付可转为制式表格。

- 上传与任务编排
  - 验收口径：支持单/多段视频上传；提交后生成任务ID；状态可查询；失败可重试
  - 度量指标：任务成功率≥99%；P95入队耗时≤3s；重试成功率≥95%
  - 数据来源：队列日志、任务结果目录、监控面板
  - 备注：限制单文件大小，自动压缩与转码
- 自动回合切分
  - 验收口径：与人工标注对齐率≥90%（样例集）
  - 度量指标：F1≥0.90；平均边界偏差≤±10帧
  - 数据来源：标注对照集、评测脚本
  - 备注：阈值与最小时长可配置
- 左右击剑手分侧建模
  - 验收口径：结果页可区分 Fencer_Left/Fencer_Right 并分别统计
  - 度量指标：分侧混淆率≤2%
  - 数据来源：回放抽查、JSON 字段校验
- 指标与图表
  - 验收口径：生成八维雷达、趋势、标签分布、按视频统计；空数据有兜底提示
  - 度量指标：渲染成功率≥99%；PNG分辨率≥300DPI；中文字体正确渲染
  - 数据来源：结果目录、前端截图、渲染日志
- 标签与检索
  - 验收口径：支持回合级标签增删改查、侧别绑定与条件筛选
  - 度量指标：CRUD 成功率≥99%；索引命中时间≤200ms（本地DB）
  - 数据来源：DB 查询日志与页面交互记录
- 报告导出与归档
  - 验收口径：支持 CSV/JSON/PNG、ZIP 打包下载；历史可检索
  - 度量指标：下载成功率≥99%；平均打包耗时≤10s（常规规模）
  - 数据来源：Nginx/应用日志、下载流水
- 安全与隐私
  - 验收口径：登录鉴权、目录隔离、HTTPS/反代可选、敏感配置外置
  - 度量指标：越权访问阻断率100%；弱口令拦截率100%
  - 数据来源：渗透与越权用例、配置审计
- 运维观测与备份
  - 验收口径：提供CPU/GPU/内存/磁盘/队列/失败率监控与告警；提供周/月度备份
  - 度量指标：告警平均响应≤15min；备份成功率≥99%
  - 数据来源：监控平台、备份日志

### 3. 典型使用流程

#### 3.1 账号登录与权限校验
- 输入：用户名、密码；会话Cookie/Token
- 校验：账号存在、密码强度、登录频次限制、防暴力尝试
- 处理模块：Flask-Login 会话管理，权限守卫（仅访问本人 results 子树）
- 产出：登录态、导航至上传页
- 边界：多端登录、过期会话、弱口令；提供重置与封禁策略
- 指标：登录成功率、平均登录耗时、越权阻断率

#### 3.2 视频上传与多段合并
- 输入：单段或多段视频文件；参数：武器类型、左右侧、可选跟踪索引
- 校验：文件类型/大小、重复文件哈希、路径白名单
- 处理模块：上传接收、顺序校验（多段）、生成任务ID、入队排程
- 产出：`Uploads/` 写入原始文件；`results/{user}/{upload}` 预建目录结构
- 边界：断点续传、网络抖动、大文件；自动重试与提示
- 指标：P95入队耗时、上传失败率、平均文件大小

#### 3.3 元数据提取与转码压缩
- 输入：原始视频路径
- 校验：可读性、编码格式、分辨率/码率阈值
- 处理模块：ffprobe 获取分辨率、时长、码率；ffmpeg 按策略转码至 1280x720、1500–5000kbps
- 产出：规范化视频（必要时）、元数据JSON
- 边界：异常编码、音轨异常；回退保留原文件
- 指标：转码命中率、平均转码耗时、码率/体积下降比

#### 3.4 目标检测（YOLO）
- 输入：规范化视频帧流
- 校验：模型文件可用、置信阈值配置
- 处理模块：YOLO 推理，输出人形目标框与置信度
- 产出：候选检测序列（按帧）
- 边界：光照剧变、远景、遮挡；置信动态调节
- 指标：平均推理FPS、检测召回/精度（抽样集）

#### 3.5 跟踪与分割（Track-Anything/XMem）
- 输入：检测框、首帧交互/自动初始化、历史轨迹
- 校验：左右侧绑定一致性、掩膜质量阈值
- 处理模块：分割与跟踪、掩膜膨胀/裁剪、断连重启
- 产出：左右击剑手掩膜、边界框、轨迹序列
- 边界：换面、衣物相似、长期遮挡；调用 Torchreid 辅助重识别
- 指标：跟踪中断率、重新关联成功率、显存峰值

#### 3.6 轨迹提取与信号处理
- 输入：掩膜与边界框序列
- 校验：前脚关键点可见性与置信
- 处理模块：关键点/几何中心计算；Savitzky–Golay 平滑；单位归一
- 产出：x(t)、v(t)、a(t) 序列
- 边界：关键点缺失；采用插值与回退至像素域相对指标
- 指标：轨迹覆盖率、插值占比、平滑窗口命中率

#### 3.7 自动回合切分
- 输入：x(t)、v(t)、a(t)；暂停候选
- 校验：阈值与最小时长配置（按武器类型）
- 处理模块：三态判定（前进/暂停/后退）；进攻触发、暂停终止；短停合并、过长回合分裂
- 产出：`Bout{start_frame, end_frame}` 列表与触发证据
- 边界：镜头切换、极短回合、越界；输出"待复核"标识
- 指标：与人工标注对齐 F1、平均边界偏差、回合数准确性

#### 3.8 指标计算与聚合
- 输入：回合切分结果与轨迹特征
- 校验：空回合与异常值剔除
- 处理模块：速度/加速度均值与峰值、前进/暂停比、首步滞后、伸臂频次与时长、冲刺成功、攻击比率等
- 产出：回合级 metrics 与选手汇总 metrics；JSON/CSV 明细
- 边界：无事件回合→置0并标注原因
- 指标：指标覆盖率、异常率、计算耗时

#### 3.9 可视化渲染与报告
- 输入：metrics、标签、胜负
- 校验：中文字体与配色、PNG分辨率
- 处理模块：雷达、趋势、标签分布、按视频统计图表；综合报告合成
- 产出：`fencer_analysis/advanced_plots/*` PNG；ZIP/CSV/JSON 报告
- 边界：空数据图→友好占位；大批量渲染→分批写盘
- 指标：渲染成功率、P95渲染耗时、图表点击率

#### 3.10 标签与胜负管理
- 输入：回合ID、侧别、标签名/胜负结果
- 校验：标签合法性、侧别一致性
- 处理模块：标签CRUD、胜负录入/修改；索引与筛选
- 产出：BoutTag 记录、回合结果更新
- 边界：标签冲突→提示；批量打标→事务保障
- 指标：标签使用频次、筛选命中率、冲突率

#### 3.11 导出、归档与复盘闭环
- 输入：选择的回合范围/选手/视频
- 校验：导出路径白名单、配额与容量
- 处理模块：CSV/JSON/PNG/视频剪辑导出、ZIP 打包；历史归档与查询
- 产出：下载包、报告链接、归档索引
- 边界：大包分片、网络中断断点；审计记录
- 指标：下载成功率、平均打包耗时、复盘完成率

#### 3X. 实施细则（对应3.2–3.11）

- 3.2 上传与多段合并
  - 操作步骤：选择文件→按比赛顺序添加→选择武器/侧别/索引→提交
  - API/模块：`POST /upload`，入队器（Celery）
  - 参数与默认值：max_size=500MB、allowed_ext=mp4/avi/mov、retries=3
  - 日志与监控：上传P95、失败率、队列入队耗时；记录file_hash与user_id
  - 验收与测试用例：大中小文件各3个、网络抖动模拟、重复文件拒绝
  - 常见故障与处置：超大文件→提示并压缩；网络中断→断点续传/重试

- 3.3 元数据与转码
  - 操作步骤：后端自动调用ffprobe→判断阈值→必要时执行ffmpeg
  - API/模块：`ffprobe`、`ffmpeg` 封装
  - 参数与默认值：target=1280x720、bitrate=1500–5000kbps、preset=veryfast
  - 日志与监控：转码命中率、平均耗时、出错栈
  - 验收与测试用例：不同分辨率/码率样例各2个；异常编码1个
  - 常见故障与处置：无编解码器→保留原文件并预警

- 3.4 目标检测（YOLO）
  - 操作步骤：加载权重→按帧推理→输出框与置信
  - API/模块：Ultralytics YOLO v8.3
  - 参数与默认值：conf=0.25、iou=0.45、classes=[person]
  - 日志与监控：FPS、推理耗时分布、GPU使用率
  - 验收与测试用例：公开样例与自建样例各N段，抽样评测mAP
  - 常见故障与处置：光照差→动态阈值；远景→切ROI或放宽阈值

- 3.5 跟踪与分割
  - 操作步骤：首帧初始化→逐帧更新掩膜→断连重启
  - API/模块：Track-Anything/XMem、Torchreid
  - 参数与默认值：iou_th=0.4、reid_sim=0.8、lost_limit=30帧
  - 日志与监控：中断率、重连成功率、显存峰值
  - 验收与测试用例：遮挡/换面/相似服装三类片段各2个
  - 常见故障与处置：混淆→人工复核标注并重算

- 3.6 轨迹与信号
  - 操作步骤：提取关键点→平滑→归一→导出x/v/a
  - API/模块：关键点/几何中心计算、SG滤波
  - 参数与默认值：window=11、order=3
  - 日志与监控：轨迹覆盖率、插值占比
  - 验收与测试用例：缺帧/抖动/低帧率样例
  - 常见故障与处置：关键点缺失→插值或回退像素域

- 3.7 回合切分
  - 操作步骤：三态判定→触发/终止→合并/分裂→输出证据
  - API/模块：切分器（内部模块）
  - 参数与默认值：pause_v=0.05、min_dur=0.6s、merge_gap=0.25s
  - 日志与监控：对齐F1、平均偏差、回合数
  - 验收与测试用例：与人工标注对照集（不少于30段）
  - 常见故障与处置：镜头切换→标记"待复核"并提示

- 3.8 指标与聚合
  - 操作步骤：逐回合计算→异常剔除→汇总统计
  - API/模块：metrics 计算器、analysis_builders
  - 参数与默认值：arm_ext_min=0.2s、lunge_acc=1.5
  - 日志与监控：指标异常率、计算耗时
  - 验收与测试用例：含无事件回合、极端速度回合的样例
  - 常见故障与处置：无事件→置0并输出原因

- 3.9 可视化与报告
  - 操作步骤：渲染雷达/趋势/分布/统计→合成报告→写盘
  - API/模块：Matplotlib/Seaborn、报告生成器
  - 参数与默认值：dpi=300、font=SimHei/DejaVu
  - 日志与监控：渲染耗时、图表失败率
  - 验收与测试用例：空数据/多回合/多视频组合
  - 常见故障与处置：无字体→安装中文字体并重试

- 3.10 标签与胜负
  - 操作步骤：单/批量打标、筛选、胜负录入/修改
  - API/模块：Tag/BoutTag CRUD、过滤器
  - 参数与默认值：标签长度≤32、侧别=left/right
  - 日志与监控：标签增长率、冲突率
  - 验收与测试用例：并发打标、冲突与回滚
  - 常见故障与处置：冲突→提示并合并/撤销

- 3.11 导出与归档
  - 操作步骤：选择范围→导出CSV/JSON/PNG/剪辑→ZIP → 下载
  - API/模块：导出器、打包器
  - 参数与默认值：zip_max=2GB、chunk_size=128MB
  - 日志与监控：下载成功率、平均打包时长
  - 验收与测试用例：大包/断网/断点续传
  - 常见故障与处置：磁盘不足→清理策略与限流

### 4. 系统架构
- 表层（UI）：`templates/*.html` 基于 Bootstrap 的简洁界面，中文优先；
- 应用层：`app.py` 提供路由、认证、任务编排、报告生成与下载；
- 任务层：`celery_config.py` + Redis 实现异步分析与失败重试；
- 分析层：`your_scripts/video_analysis.py`、`analysis_builders.py` 等负责检测、跟踪、切分与指标聚合；
- 数据层：`models.py` 使用 SQLAlchemy 建模（User/Fencer/Upload/Bout/Tag）；默认 SQLite，可切换 MySQL；
- 媒体工具：FFmpeg/ffprobe 负责转码、元数据与片段生成；
- 结果存储：`results/{user}/{upload}/` 树状目录，含 fencer_analysis、match_analysis、csv 等子目录。

性能与扩展：支持 GPU 加速（CUDA 12.1 + PyTorch 2.5.1），CPU 路径可用；分析与图表模块采用"插件化"设计，便于新增指标维度。

#### 4.1 总体组成与建设内容
- 应用子系统：
  - 账号与权限：登录、会话、目录隔离；
  - 上传与任务：多段合并、转码压缩、异步排队、进度查询；
  - 结果与报告：回合列表、图表预览、导出下载、综合报告；
  - 标签与检索：侧别标签、回合标注、筛选与统计；
- 算法子系统：
  - 目标检测：YOLO；
  - 跟踪与分割：Track-Anything/XMem；
  - 回合切分：基于轨迹与三态序列（前进/暂停/后退）；
  - 指标计算：速度、加速度、首步、伸臂、暂停/前进比、攻击/冲刺；
  - 图表渲染：雷达、趋势、标签分布、按视频统计；
- 数据子系统：
  - 模型与表：User/Fencer/Upload/UploadVideo/Bout/Tag/BoutTag/HolisticAnalysis；
  - 文件产物：JSON/CSV/PNG/视频片段；
  - 目录规范：`results/{user}/{upload}/...` 与清理/备份策略；
- 运行支撑：
  - 中间件：Redis、FFmpeg；
  - 反代与安全：Nginx/TLS（可选）；
  - 监控备份：容量、耗时、失败率、定期备份与恢复演练。

#### 4.2 功能子系统职责与接口
- 账号与权限
  - 职责：用户注册/登录、会话维持、访问控制；
  - 输入/输出：用户名、密码 → 会话票据；
  - 关键接口：`/login`、`/logout`、守卫装饰器；
- 上传与任务
  - 职责：视频上传、参数校验、转码压缩、任务入队与追踪；
  - 输入/输出：视频文件、武器、侧别、索引 → 任务ID、进度、结果路径；
  - 关键接口：`/upload`、`/status`；
- 分析与图表
  - 职责：检测/跟踪/切分/指标/渲染；
  - 输入/输出：视频与参数 → JSON/CSV/PNG/剪辑；
  - 模块接口：`process_first_frame`、`TrackingAnything`、`analysis_builders`；
- 结果与报告
  - 职责：浏览、筛选、下载、综合报告；
  - 输入/输出：查询条件 → 列表、图表、ZIP 包；
  - 关键接口：`/results`、`/download`、`/report`。

#### 4.3 算法引擎组成与数据流
1) 检测：YOLO 按帧检测人体 → 置信度与框坐标；
2) 跟踪/分割：Track-Anything/XMem 生成掩膜与轨迹 → 左/右侧绑定；
3) 轨迹与三态推断：前脚 x(t) → v(t)/a(t) → 前进/暂停/后退序列；
4) 回合切分：进攻触发、暂停终止、阈值与最小时长；
5) 指标计算：回合级与汇总级；
6) 可视化：雷达/趋势/分布/统计；
7) 产物入库：JSON/CSV/PNG/片段写入 results 目录。

#### 4.4 数据资源组成
- 原始视频：训练与比赛素材；
- 结构化数据：回合 JSON、选手指标、标签与胜负；
- 可视化产物：PNG 图表、剪辑视频；
- 元数据：分辨率、帧率、码率、耗时与日志；
- 口径与版本：指标参数、模型权重版本与哈希。

#### 4.5 运行支撑与安全保障
- 资源：CPU/GPU、内存与磁盘配额、上传限额；
- 中间件：Redis、数据库（SQLite/MySQL）、FFmpeg；
- 安全：会话鉴权、最小权限、HTTPS/反代、日志脱敏、备份加密；
- 运维：容量/耗时/失败率监控、队列水位告警、备份恢复演练。

#### 4.6 管理与合规保障
- 角色与职责：产品、教练/教师、运维、算法、测试；
- 合规：未成年人保护、肖像/著作权、用途限制与保密；
- 审计与追溯：请求与导出日志、版本与参数留痕；
- 绩效：使用率、复盘时效、指标改善度、问题闭环率。

#### 4A. 接口清单（路由/方法/参数/示例）

- 上传任务
  - 路由：`POST /upload`
  - 参数（form-data）：`file`(必填)、`weapon`(saber/foil/epee)、`left_fencer_id`、`right_fencer_id`、`selected_indexes`
  - 响应：`{ task_id, upload_id }`
  - 示例：curl -F "file=@match1.mp4" -F "weapon=saber" http://host/upload

- 任务状态查询
  - 路由：`GET /status`
  - 参数（query）：`task_id`
  - 响应：`{ status: pending|running|done|error, progress, message }`

- 结果列表
  - 路由：`GET /results`
  - 参数：`upload_id`
  - 响应：结果目录树与可下载项列表

- 下载文件
  - 路由：`GET /download`
  - 参数：`path`（仅允许 results 子树）
  - 响应：文件流（支持 Range）

- 击剑手管理
  - 路由：`GET/POST /fencers`
  - 参数：增/改需 `name`，可选 `user_id`
  - 响应：列表或变更结果

- 综合报告生成
  - 路由：`POST /report`
  - 参数：`fencer_id`、`range`（可选：upload_id 列表或时间范围）
  - 响应：`{ report_path, status }`

- 回合胜负录入
  - 路由：`POST /select-winners`
  - 参数：`bout_id`、`result`(left|right|skip)
  - 响应：更新后的回合对象

- 标签 CRUD
  - 路由：`POST /tags`、`DELETE /tags/{id}`、`GET /tags`
  - 参数：`bout_id`、`fencer_side`(left|right)、`name`
  - 响应：标签对象/列表

- 认证相关
  - 路由：`POST /login`、`GET /logout`、`POST /register`
  - 参数：用户名、密码
  - 响应：会话/错误信息

安全约束：除 `/login`、`/register` 外均需登录；`/download` 路径限制在 `results` 子树；CSRF 防护启用。

#### 4B. 数据字典（JSON/CSV 字段说明）

- 回合分析 JSON（示例：`match_analysis/*_analysis.json`）
  - 顶层字段：`match_idx`(int)、`video_path`(str)、`winner_side`(left|right|undetermined)
  - `left_data`/`right_data`：
    - `interval_analysis.advance_analyses[]`：`attack_info.has_attack(bool)`,`attack_type(str)`,`tempo_type(str)`,`avg_distance(float)`
    - `retreat_analyses[]`：`counter_opportunities[]`、`opportunities_taken(int)`、`opportunities_missed(int)`
    - `movement_data`：`advance_intervals[]`、`retreat_intervals[]`、`pause_intervals[]`
  - `metrics`（回合级）：`velocity(float)`、`acceleration(float)`、`advance_ratio(float)`、`pause_ratio(float)`、`first_step_init(float)`、`arm_extension_freq(int)`、`avg_arm_extension_duration(float)`、`has_launch(bool)`、`is_attacking(bool)`

- 选手汇总 JSON（示例：`fencer_Fencer_Left_analysis.json`）
  - `fencer_id`(str)、`metrics`（汇总）：`total_bouts(int)`、`avg_velocity(float)`、`avg_acceleration(float)`、`avg_advance_ratio(float)`、`avg_pause_ratio(float)`、`avg_first_step_init(float)`、`total_arm_extensions(int)`、`avg_arm_extension_duration(float)`、`launch_success_rate(float)`、`attacking_ratio(float)`
  - `bouts[]`：包含回合级 `metrics` 与区间明细

- CSV 明细（典型列）
  - `user_id, upload_id, bout_id, side(left/right), match_idx, start_frame, end_frame, velocity, acceleration, advance_ratio, pause_ratio, first_step_init, arm_extension_freq, avg_arm_extension_duration, has_launch, is_attacking, result`
  - 说明：时间以秒/帧并列（如有）；比率0–1；布尔为true/false；编码UTF-8

- 图表产物命名规范
  - 路径：`results/{user}/{upload}/fencer_analysis/advanced_plots/Fencer_Left/left_bout_outcome.png` 等
  - 规则：包含 `Fencer_Left|Right`、图表类型、回合或汇总标识

- 下载白名单
  - 仅允许 `results` 子树内：`*.csv`、`*.json`、`*.png`、`*.zip`，防止目录穿越。

### 5. 数据模型（核心实体）
- User：账户体系；
- Fencer：击剑手档案，作为左右侧绑定；
- Upload：一次上传任务（可为多段合并），包含武器类型、进度与产物路径；
- UploadVideo：多段视频元素，维护序号与偏移；
- Bout：回合单元，记录起止帧、扩展视频与胜负结果；
- Tag/BoutTag：标签体系，支持侧别与回合级标注；
- HolisticAnalysis：综合报告记录（可扩展）。

### 6. 关键算法与指标口径
6.1 检测与跟踪
- YOLO 进行行人检测；Track-Anything/XMem 对左右击剑手进行分割与跟踪；
- Torchreid（可选）进行重识别辅助；异常帧、遮挡与镜头切换包含兜底策略。

6.2 自动回合切分
- 基于前脚轨迹与速度/加速度的三态（前进/暂停/后退）序列；
- 以"进攻状态触发、暂停阈值终止"为主规则，辅以去噪、滑窗与最小时长约束。

6.3 指标集合（每回合与汇总）
- 速度与加速度（平均值、峰值）；
- 前进/暂停比率；
- 首步启动滞后（first_step_init）；
- 伸臂次数与平均时长；
- 冲刺成功（has_launch 比例）；
- 攻击比率（处于进攻状态时间占比）；
- 胜负与右路权相关得分（若可用）。

6.4 雷达图与趋势图
- 八维雷达：速度、加速度、前进比率、暂停控制（1-暂停比率）、首步、伸臂、冲刺成功、攻击比率；
- 趋势图：跨回合速度与前进比率；
- 标签分布图与回合摘要图；
- 可视化在中文字体环境下渲染（SimHei/DejaVu Sans 等），保存为 300DPI PNG。

#### 6A. 算法详述：输入—处理—参数—边界—复杂度—失败路径—证据
- 自动回合切分
  - 输入：前脚轨迹 x(t)、速度 v(t)、加速度 a(t)、暂停区间候选
  - 处理：Savitzky–Golay 平滑 → 三态判定（前进/暂停/后退）→ 进攻触发/暂停终止 → 合并/分裂（t_min、gap）
  - 参数：窗口=11、阶数=3、暂停阈值 ε_v、最小时长 t_min（按武器类型微调）
  - 边界：短回合、镜头切换、遮挡、极低帧率；均有兜底与回退规则
  - 复杂度：O(N) 单次扫描
  - 失败路径：置信不足→标记"待复核"；保留证据（片段、速度曲线PNG）
  - 证据：区间叠加图、事件时间戳、对齐评分
- 左右侧跟踪与重识别
  - 输入：YOLO 框、分割掩膜、历史轨迹
  - 处理：掩膜中心/几何约束+相似度匹配；Torchreid 作为断连回路
  - 参数：IoU/相似度阈值、丢失帧上限
  - 边界：遮挡/换面/服装相似
  - 复杂度：O(NK)（K为匹配对数）
  - 失败路径：断连→重启跟踪；左右混淆→回放审查
  - 证据：框轨迹叠加、掩膜快照
- 指标计算与归一
  - 输入：切分后的回合序列与轨迹
  - 处理：速度/加速度/比率/首步/伸臂/冲刺/攻击；标准化至0–1（雷达）
  - 参数：伸臂时长阈值、冲刺判定阈值
  - 边界：无事件回合（输出0并提示）
  - 复杂度：O(N)
  - 证据：指标表、回合摘要、趋势图

### 7. 用户体验（UX）与交互
- 上手路径：上传 → 状态页 → 结果页 → 报告导出；
- 结果页：回合列表、侧别切换、图表预览、标签操作与胜负录入；
- 可导出 CSV/JSON/PNG/剪辑视频；
- 未命中数据场景具备友好占位提示；
- 账号与结果目录按用户隔离，支持历史回溯与归档。

### 8. 权限、安全与隐私
- 登录鉴权与会话控制；最小权限访问 `results` 子树；
- 默认不外发第三方服务；部署可选本地/私有云；
- 审计日志记录关键操作（上传、分析、导出）；
- 敏感配置（Key/DB 凭据）采用环境变量；
- 可选 HTTPS/Nginx 反代与访问控制策略；
- 数据留存与备份策略可按单位规范配置。

### 9. 部署与运维
- 依赖：Python 3.11、Redis、FFmpeg、PyTorch（GPU 环境可选）；
- 数据库：默认 SQLite，可配置 MySQL；
- 服务：Flask 应用 + Celery Worker；
- 反向代理：Nginx（可选），启用 TLS；
- 目录权限：`Uploads/` 与 `results/` 需可写；
- 监控：关注队列长度、处理耗时、GPU/CPU/IO 指标，任务失败率与磁盘容量；
- 备份：结果与数据库按周期归档；
- 容器化/编排：可扩展至 Docker/K8s。

### 9A. 运维与安全细化
- 监控指标与阈值（建议起点）
  - CPU>85%（5m）告警；内存>85%；磁盘>80%；GPU显存>90%
  - 队列长度>阈值（按并发×任务耗时估算）；任务失败率>1%；P95耗时异常
  - 下载失败率>1%；Nginx 4xx/5xx 异常上升
- 告警与响应
  - 渠道：邮件/IM；SLA：P1≤15min，P2≤2h
  - 值班：排班表+应急通讯录
- 备份与恢复SOP
  - 周期：数据库日增量、周全量；结果目录日增量
  - 演练：季度恢复演练，记录RTO/RPO
  - 校验：抽样校验哈希与可读性
- 变更与发布
  - 流程：预生产验证→灰度→全量；有回滚脚本与窗口期
- 安全基线
  - 账号策略（强口令、过期、锁定）、最小权限、HTTPS/TLS、日志脱敏
  - 依赖扫描（SAST/DAST）、镜像签名与SBOM（可选）

### 9B. 故障处理SOP（摘录）
- 分级：P1（不可用/数据风险）、P2（性能退化）、P3（一般问题）
- 止血：限流/扩容/回滚；保留问题现场（日志、任务样本）
- 根因定位：按链路分层排查（队列→算法→IO→前端）
- 复盘：输出 5 Whys、行动项与跟踪编号

### 10. 性能、稳定性与成本
- GPU 环境可实现接近实时的处理速度；
- 大文件自动转码与降分辨率控制带宽与存储；
- 异步队列与重试机制提升稳健性；
- 可按预算选择 CPU 或 GPU 路径，按需扩容。

### 11. 可视化与报告样例
- 产物路径：`results/{user}/{upload}/fencer_analysis/advanced_plots/Fencer_Left/` 与 `Fencer_Right/`；
- JSON 示例：`fencer_Fencer_Left_analysis.json`，包含 `metrics` 与 `bouts` 明细；
- 报告可合成"个人雷达 + 回合趋势 + 标签分布 + 按视频统计"的合集图，便于教学与汇报。

### 12. 兼容与扩展
- 武器：saber/foil/epee 不同阈值与口径可配置；
- 算法：检测/跟踪/重识别模块可替换与升级；
- 数据：导入既有标注或第三方分析数据进行对照；
- 输出：支持更多格式（PDF 报告、打包 ZIP 等）。

### 12A. 产品组成清单与项目组织保障
- 硬件与环境（建议规模）
  - 校园训练版：8核CPU、32GB内存、GPU≥8GB显存、SSD≥1TB
  - 俱乐部/赛事版：16核CPU、64GB内存、1–2块GPU、SSD/NAS≥4TB
- 软件与中间件
  - Python 3.11、PyTorch/CUDA、Ultralytics、MMCV、FFmpeg、Redis、Nginx、SQLite/MySQL
- 交付内容
  - 源码/镜像、部署脚本、使用手册、运维SOP、训练样例、测试用例、汇报模板
- 项目组织与治理
  - 角色：产品（需求/口径）、算法（模型与指标）、后端（服务与数据）、前端（交互展示）、运维（监控与备份）、测试（质量门禁）
  - 流程：需求→设计→实现→评审→联调→演示→验收→运维交接→培训
  - 里程碑：Alpha（核心链路）→Beta（全功能与稳定）→RC（性能与安全）→GA（交付与运营）
  - 质量门禁：代码评审、CI校验、用例通过率、性能阈值、渗透与越权用例通过
  - 绩效与汇报：周进展、风险台账、关键指标看板

### 13. 路线图（摘要）
- v1.x：稳固核心链路与可视化；
- v2.0：增加在线微调与更丰富的战术事件识别；
- 教学生态：案例库、对比计划与课堂模式；
- 运维：监控看板与告警、自动化备份复原。

### 14. 常见问题（FAQ）
- Q：CPU 能跑吗？A：支持，速度较慢；推荐 GPU。
- Q：结果为空怎么办？A：视频过短或无有效回合；可调整阈值或改用更清晰素材。
- Q：如何录入回合胜负？A：在结果页选择"胜负录入/修改"。
- Q：如何批量导出？A：在结果页选择范围导出 CSV/JSON/PNG（或 ZIP）。

### 15. 附录
- 代码入口：`app.py`
- 数据模型：`models.py`
- 分析主流程：`your_scripts/video_analysis.py`
- 指标构建器：`your_scripts/analysis_builders.py`
- 结果样例：`results/1/159/fencer_analysis/`
- 依赖清单：`requirements.txt`

### 附加A · 2A表格版（含基线指标）

| 需求条目 | 验收口径 | 度量指标(基线) | 数据来源 | 备注 |
|---|---|---|---|---|
| 上传与任务编排 | 单/多段上传；任务ID；状态查询；失败可重试 | 成功率≥99%；P95入队≤3s；重试成功率≥95% | 队列/应用日志、监控 | 大文件自动压缩与转码 |
| 自动回合切分 | 与人工标注对齐率≥90% | F1≥0.90；平均偏差≤±10帧 | 评测脚本+标注集 | 阈值/窗口可配 |
| 分侧建模 | 左/右侧指标独立输出 | 混淆率≤2% | JSON校验、抽查 | 断连重识别回路 |
| 指标与图表 | 雷达/趋势/分布/统计；空数据兜底 | 渲染成功率≥99%；PNG≥300DPI | 结果目录、渲染日志 | 中文字体正确 |
| 标签与检索 | 回合级CRUD与筛选 | CRUD成功率≥99%；索引≤200ms | DB日志 | 字典可配 |
| 报告导出与归档 | CSV/JSON/PNG/ZIP | 下载成功率≥99%；打包≤10s | Nginx/应用日志 | ZIP白名单路径 |
| 安全与隐私 | 登录/目录隔离/HTTPS/外置配置 | 越权拦截100%；弱口令覆盖100% | 渗透/越权用例 | 审计就绪 |
| 运维与备份 | 监控与周/月备份 | 告警响应≤15min；备份成功≥99% | 监控平台、备份日志 | 恢复演练季度 |

基线说明：基线为起步值，随部署规模、硬件条件与赛季峰值动态调整；建议在试运行阶段每两周复核一次。

### 附加B · 6A参数默认值与可配置项

| 模块 | 参数 | 默认值 | 说明 | 可调范围 |
|---|---|---|---|---|
| 平滑 | sg_window | 11 | Savitzky–Golay窗口 | 奇数5–21 |
| 平滑 | sg_order | 3 | 多项式阶数 | 2–4 |
| 三态判定 | pause_v_epsilon | 0.05 | 速度阈值(归一) | 0.02–0.1 |
| 回合 | min_bout_duration | 0.6s | 最小时长 | 0.3–1.2s |
| 回合 | merge_gap | 0.25s | 间隙合并阈值 | 0.1–0.6s |
| 跟踪 | iou_threshold | 0.4 | 匹配IoU阈值 | 0.3–0.6 |
| 跟踪 | reid_sim_threshold | 0.8 | 重识别相似度阈值 | 0.7–0.9 |
| 跟踪 | lost_frame_limit | 30帧 | 断连上限 | 10–60帧 |
| 指标 | arm_ext_min | 0.20s | 伸臂最小时长 | 0.1–0.4s |
| 指标 | lunge_acc_threshold | 1.5 | 冲刺加速度阈值 | 1.0–3.0 |
| 图表 | radar_scale | 0–1 | 雷达归一范围 | 固定 |
| 压缩 | max_resolution | 1280x720 | 压缩目标分辨率 | 960x540–1920x1080 |
| 压缩 | max_bitrate_kbps | 5000 | 最大码率 | 1500–8000 |

可配置方式：.env/配置文件；按武器类型(saber/foil/epee)设差异阈值；前端提供"分析参数模板"预设（保守/标准/进攻型/防守型）。

### 附加C · 9C运维仪表盘与备份恢复演练模板

- 仪表盘指标定义（建议Prometheus导出）
  - 系统：cpu_usage、mem_usage、disk_usage、net_io
  - GPU：gpu_mem_used、gpu_util、gpu_temp
  - 队列：queue_depth、task_started_total、task_failed_total、task_latency_p95
  - 应用：analyze_duration_p95、download_error_rate、render_success_rate
  - Web/Nginx：http_4xx_rate、http_5xx_rate、request_duration_p95
  - 存储：results_dir_size、backups_success_total、backups_duration
- 阈值示例：
  - cpu_usage>85%(5m)、mem_usage>85%、disk_usage>80%、gpu_mem_used>90%
  - queue_depth>并发×平均耗时/时间窗、task_failed_rate>1%
  - http_5xx_rate>0.5% 或 request_duration_p95 持续上升
- 看板建议：总览、任务链路、GPU健康、下载与导出、备份恢复、容量趋势

- 备份恢复演练模板（摘录）
  - 目标：在RTO≤4小时、RPO≤24小时内恢复应用与关键数据
  - 周期：季度
  - 范围：SQLite/MySQL数据库、results目录关键产物
  - 步骤：
    1) 选取最近一次全量+若干增量备份；
    2) 在隔离环境还原数据库与结果目录；
    3) 校验：抽样比对JSON/CSV/PNG哈希与内容可读性；
    4) 回放：随机任务的结果页与下载链路；
    5) 记录：用时、问题与改进项（表单模板）；
    6) 复盘会议：形成行动项并跟踪闭环。
  - 记录表（关键字段）：演练编号、日期、参与人、备份集版本、RTO/RPO、差错明细、纠正措施、下次改进目标。

### 附加D · 用户价值详解（对应第2节）

- 分角色价值
  - 教练/教师：
    - 训练处方：依据雷达短板（如首步滞后、暂停过长）制定每周训练计划；
    - 课堂范例库：从结果目录抽取"典型错误/优秀动作"作为课堂演示素材；
    - 客观评估：期初/期中/期末三次测评曲线，减少主观争议；
    - 班级/队内管理：按标签统计学员常见问题，批量生成作业清单；
  - 运动员：
    - 个人成长：速度/节奏/伸臂/冲刺等曲线可视化，对比"本周vs上周""赛前vs赛后"；
    - 自主复盘：下载回合片段+标签清单，自评与教练评估对齐；
    - 目标激励：针对单一指标（如暂停控制）设定提升目标与周挑战；
  - 俱乐部/队伍管理：
    - 训练质量：课程覆盖率、复盘完成率、指标整体提升度；
    - 招生与宣传：可发布的图表与成长故事；
    - 科研与合作：结构化数据支持课题与论文复现；
  - 裁判/赛事技术：
    - 赛后复盘：关键回合证据（片段+曲线+标签），支撑复审与培训；

- 场景化流程（示例）
  1) 课前：教练创建"本周训练短板指标"模板（如首步、暂停控制）。
  2) 课后：学员上传训练视频，系统自动分析并生成个人雷达与回合摘要。
  3) 复盘：教练批注关键回合，打上"时机过早/距离过近"等标签；
  4) 处方：生成"一周训练处方"（频次/组数/指标目标），下周自动对比；
  5) 汇报：月底自动汇总班级与个人进步率，生成 PPT 素材。

- 量化指标与ROI（示例口径）
  - 复盘效率：人均每场复盘从60min降至≤25min（-58%）；
  - 教学覆盖：每周复盘人数≥80%，P50指标提升≥15%/月；
  - 训练质量：暂停比率下降≥10%；首步启动滞后下降≥20%；
  - 赛事准备：关键回合库命中率≥70%，赛后复盘出具时效≤24h；
  - 管理运营：可发布内容产出≥4条/月，家校沟通满意度≥90%。

- 采纳路线（建议）
  - 第1月（试运行）：10–20个样例视频，建立口径与基线；
  - 第2月（推广）：覆盖全员训练，形成"处方-对比-复盘"闭环；
  - 第3月（固化）：按班级/小队设置指标看板和月报；
  - 第4月+（深化）：扩展赛事复盘、科研输出与对外宣传应用。

#### 4C. 请求/响应示例（JSON）

- 示例：创建上传任务（响应包含 task_id 与 upload_id）
```json
{
  "task_id": "celery-2f9a...",
  "upload_id": 159
}
```

- 示例：查询任务状态
```json
{
  "status": "running",
  "progress": 62,
  "message": "tracking in progress"
}
```

- 示例：结果列表（节选）
```json
{
  "upload_id": 159,
  "paths": [
    "results/1/159/fencer_analysis/fencer_Fencer_Left_analysis.json",
    "results/1/159/fencer_analysis/advanced_plots/Fencer_Left/left_bout_outcome.png"
  ]
}
```

- 示例：下载受限路径错误
```json
{
  "error": "path_not_allowed",
  "message": "Download path must be under results directory"
}
```

#### 4D. 错误码规范与示例

| 错误码 | HTTP | 含义 | 建议处理 |
|---|---|---|---|
| invalid_params | 400 | 参数缺失或不合法 | 检查必填项与格式 |
| unauthorized | 401 | 未登录或会话失效 | 重新登录 |
| forbidden | 403 | 越权访问或路径不允许 | 检查权限/白名单 |
| not_found | 404 | 资源不存在 | 校验 upload_id/bout_id |
| conflict | 409 | 重复提交或状态冲突 | 幂等重试/刷新状态 |
| too_large | 413 | 文件过大 | 压缩或降低分辨率 |
| rate_limited | 429 | 触发限流 | 稍后重试 |
| server_error | 500 | 内部错误 | 查看日志/工单 |
| task_failed | 502 | 任务执行失败 | 触发重试/排查依赖 |
| backend_unavailable | 503 | 中间件不可用 | 检查Redis/DB/FFmpeg |

说明：应用响应体建议统一 `{ error, message, details? }`，并在日志中记录 trace-id 便于追踪。

#### 5A. 字段字典（主要表）

- User
  - id(int, PK)、username(varchar, unique, not null)、password(varchar, not null)
- Fencer
  - id(int, PK)、user_id(FK→User.id, not null)、name(varchar, not null)
- Upload
  - id(int, PK)、user_id(FK→User.id)、video_path(varchar, nullable)、status(varchar)、weapon_type(enum: saber/foil/epee, not null, default saber)、is_multi_video(bool, default false)、match_title(varchar, nullable)、total_bouts(int)、csv_dir(varchar)、bouts_analyzed(int)
- UploadVideo
  - id(int, PK)、upload_id(FK→Upload.id, not null)、video_path(varchar, not null)、sequence_order(int, not null)、selected_indexes(varchar, nullable)、status(varchar, default pending)、total_bouts(int)、bouts_offset(int)
- Bout
  - id(int, PK)、upload_id(FK→Upload.id, not null)、upload_video_id(FK→UploadVideo.id, nullable)、match_idx(int, not null)、start_frame(int, not null)、end_frame(int, not null)、video_path(varchar, nullable)、extended_video_path(varchar, nullable)、result(enum: left/right/skip, nullable)
- Tag
  - id(int, PK)、name(varchar, unique, not null)
- BoutTag
  - id(int, PK)、bout_id(FK→Bout.id, not null)、tag_id(FK→Tag.id, not null)、fencer_side(enum: left/right, not null)
- HolisticAnalysis
  - id(int, PK)、fencer_id(FK→Fencer.id, not null)、user_id(FK→User.id, not null)、report_path(varchar, nullable)、status(varchar, default Pending)

#### 5B. 关系与约束
- User 1..N Upload、1..N Fencer；
- Upload 1..N UploadVideo、1..N Bout；
- Bout 1..N BoutTag；Tag 1..N BoutTag；
- Fencer 与 Upload 通过 left_fencer_id/right_fencer_id 参与上传上下文（见 `models.py`）；
- 约束：外键参照完整性、`Tag.name` 唯一、`sequence_order` 非空；
- 业务约束：`fencer_side` 仅取 left/right；`result` 仅取 left/right/skip。

#### 5C. 索引与性能建议
- 常用查询索引：
  - Upload(user_id, id)
  - UploadVideo(upload_id, sequence_order)
  - Bout(upload_id, match_idx)、Bout(upload_video_id)
  - BoutTag(bout_id)、BoutTag(tag_id, fencer_side)
- 批量读：结果页采用按 `upload_id` 扫描回合与标签；
- 写入策略：短事务、分批提交；
- 大表切分：不建议早期分库分表；按 `user_id` 归档即可。

#### 5D. 迁移与版本管理
- 迁移：优先使用 SQLAlchemy 自动迁移（或手工脚本）
- 版本：以语义化版本管理模型变更，并在结果JSON中写入 `schema_version`；
- 兼容：新增字段默认可空；对外导出CSV/JSON保持向后兼容；
- 回滚：提供降级脚本与数据备份。

#### 5E. 样例记录与查询（伪SQL）
```sql
-- 查询某上传的回合与标签
SELECT b.id, b.match_idx, b.start_frame, b.end_frame, bt.fencer_side, t.name
FROM bout b
LEFT JOIN bout_tag bt ON bt.bout_id = b.id
LEFT JOIN tag t ON t.id = bt.tag_id
WHERE b.upload_id = :upload_id
ORDER BY b.match_idx ASC;

-- 查询选手在多个上传中的胜率概览（伪）
SELECT u.id AS upload_id,
       SUM(CASE WHEN b.result = 'left' THEN 1 ELSE 0 END) AS left_wins,
       SUM(CASE WHEN b.result = 'right' THEN 1 ELSE 0 END) AS right_wins
FROM upload u
LEFT JOIN bout b ON b.upload_id = u.id
WHERE u.user_id = :user_id
GROUP BY u.id;
```

#### 5F. 数据留存与合规
- 留存策略：
  - 数据库：活跃数据长期；日志与中间过程按 90–180 天滚动；
  - 结果目录：按用户/上传分层，支持按项目/学期归档；
- 访问控制：按用户隔离 `results` 子树；敏感下载需审计；
- 合规与版权：建议在上传页提示素材来源与授权；遵循校内与俱乐部隐私规范；
- 备份与恢复：周全量+日增量，季度演练（参见"附加C"）。

#### 5G. 数据流转（从上传到报告）
1) 原始视频入库 → 2) 元数据提取与（可选）转码 → 3) 检测/分割/跟踪 → 4) 轨迹与三态 → 5) 回合切分 → 6) 指标计算与聚合 → 7) 可视化与报告 → 8) 归档与下载。
- 每步均记录 `{timestamp, version, params, outputs}` 至元数据，便于复现。

#### 5H. 存储布局与命名规范
- 目录：`results/{user_id}/{upload_id}/`
  - `fencer_analysis/`：`fencer_Fencer_Left_analysis.json`、`advanced_plots/Fencer_Left/*.png`
  - `match_analysis/`：`*_analysis.json`（回合粒度）
  - `csv/`：明细数据
  - `matches/`：回合剪辑视频（可选）
- 命名：包含侧别、回合编号、图表类型；全部 UTF-8，禁止空格。

#### 5I. 数据质量与校验规则
- 结构校验：JSON schema（必填字段、类型、范围）；
- 值域校验：比率0–1、非负、帧号区间合法；
- 互斥与逻辑：`start_frame < end_frame`、`pause_ratio + advance_ratio ≤ 1`
- 完整性：回合总数、标签关联、胜负录入一致性；
- 质量评分：轨迹覆盖率、插值占比、缺失事件占比；
- 校验失败：标记并输出 `quality_report.json`。

#### 5J. 数据契约与版本
- 契约：对外导出CSV/JSON字段稳定；新增字段保持向后兼容；
- 版本：在导出头部写入 `schema_version`、`generator_version`、`params_hash`；
- 破坏性变更：升级大版本并提供转换脚本。

#### 5K. 导入与导出
- 导入：
  - 既有标注（回合边界、胜负、标签）的CSV/JSON；
  - 合并策略：若与自动结果冲突，以人工导入为主并标记来源；
- 导出：
  - 明细（CSV/JSON）、图表（PNG）、报告（ZIP）；
  - 批量导出按条件筛选（时间段、标签、侧别、回合范围）。

#### 5L. 权限矩阵（摘要）
- 普通用户：访问本人 `results`、管理本人上传与标签、下载本人报告；
- 管理员（可选）：查看汇总统计与资源占用，不得访问他人原始视频内容（除非授权）；
- 审计账号（可选）：只读日志与质量报告，不含结果内容下载；
- 访客：无权限。

#### 5M. 数据血缘与审计
- 元信息：记录来源视频、任务ID、算法版本、参数、生成时间；
- 链路追踪：trace-id 贯穿上传→分析→渲染→导出；
- 审计：记录下载与共享动作（时间、用户、对象、结果）。

#### 5N. 留存、归档与清理
- 留存：活跃项目长期；历史项目按学期/年度归档；
- 归档：打包至冷存或对象存储，保留目录结构与元数据；
- 清理：超过留存期的临时产物（中间帧、缓存）按策略删除；
- 例外：带"长期保留"标签的项目不清理。

#### 5O. 错误恢复与一致性
- 失败任务重试：指数退避；
- 幂等：按 `task_id` 去重；
- 一致性：写入采用"先产物后登记"的两阶段；
- 局部失败：允许恢复至最近成功步骤并继续。

#### 5P. 端到端样例（最小闭环）
- 输入：`match_2025_schoolA_saber.mp4`（3段合并）、武器=saber、侧别=left/right；
- 核心参数：sg_window=11、pause_v=0.05、min_bout=0.6s、reid_sim=0.8；
- 输出：8个回合、雷达图2张、趋势图2张、标签分布2张、按视频统计2张；
- 质量报告：轨迹覆盖率>92%、插值<5%；
- 导出：`report_schoolA_2025Q1_left.zip`（CSV/JSON/PNG）。

#### 6A. 指标定义与公式（正式口径）

| 名称 | 定义 | 计算公式/方法 | 单位 | 适用条件 | 备注 |
|---|---|---|---|---|---|
| 平均速度(avg_velocity) | 回合内前脚x位移的一阶平均 | \( \bar v = \frac{1}{T}\int_0^T |v(t)| dt \) 或离散均值 | m/s | 轨迹覆盖≥80% | 归一化用于雷达 |
| 平均加速度(avg_acceleration) | 回合内速度变化率平均 | \( \bar a = \frac{1}{T}\int_0^T |a(t)| dt \) | m/s² | 同上 | 峰值另算 |
| 前进比率(advance_ratio) | 前进状态时长/总时长 | advance_time / T | 0–1 | 三态判定完成 | |
| 暂停比率(pause_ratio) | 暂停状态时长/总时长 | pause_time / T | 0–1 | 同上 | 雷达中展示为 1-pause |
| 首步滞后(first_step_init) | 接到信号至速度超过阈值的延迟 | t(v>v_th) - t(signal) | s | 有触发信号 | 无信号回合置NA |
| 伸臂次数(total_arm_extensions) | 伸臂动作次数 | 区间计数 | 次 | 手臂特征可见 | 最小时长阈值过滤 |
| 伸臂平均时长(avg_arm_extension_duration) | 单次伸臂持续平均 | mean(end-start) | s | 同上 | |
| 冲刺成功(launch_success_rate) | 存在有效冲刺比例 | sum(has_launch)/N | 0–1 | 回合级判定 | |
| 攻击比率(attacking_ratio) | 处于进攻状态时间占比 | attacking_time/T | 0–1 | 攻防判读完成 | |

说明：离散实现采用采样点求和；所有比率在空回合输出0并标注原因。

#### 6B. 典型参数配置（可按武器类型覆盖）
- 平滑：sg_window=11、sg_order=3；
- 三态阈值：pause_v_epsilon=0.05、advance_min=0.1s；
- 合并/分裂：merge_gap=0.25s、min_bout_duration=0.6s；
- 伸臂：arm_ext_min=0.2s；
- 冲刺：lunge_acc_threshold=1.5（m/s²）；
- 右路权相关（若启用）：事件窗口=0.4s；
- 可视化：dpi=300、font=SimHei/DejaVu。

#### 6C. 误差与不确定性评估
- 轨迹误差：关键点缺失/抖动→采用SG平滑与插值；报告轨迹覆盖率与插值占比；
- 切分误差：以人工标注为金标准，评估 F1、平均边界偏差（帧）与编号一致性；
- 指标不确定性：对回合内指标采用自助法（bootstrap）估计置信区间（如95% CI）；
- 模型不确定性：权重版本差异记录于 `params_hash`；重大变更需再标定；
- 误差传播：速度→加速度为一阶差分，放大噪声，需更强平滑或鲁棒估计。

#### 6D. 图证样例与可解释性
- 每次切分输出"区间叠加图（状态条）+速度/加速度曲线PNG"；
- 冲刺/首步事件在曲线上以标记点展示，附时刻与数值；
- 标签分布图展示"积极/需改进"分层计数；
- 雷达图中标注总回合数与各维度归一规则。

#### 6E. 验证与验收（指标层面）
- 数据集：公开样例+自建样例≥30段；
- 指标：
  - 回合切分 F1≥0.9、平均边界偏差≤±10帧；
  - 空回合占比≤5%（异常场景除外）；
  - 雷达图渲染成功率≥99%；
- 复核流程：抽样对照人工记录；异常项进入"待复核队列"。

#### 6F. 快速参考表（口径摘要）

| 指标 | 单位 | 采样域 | 归一化 | 缺失处理 | 证据 |
|---|---|---|---|---|---|
| avg_velocity | m/s | 回合/选手 | 雷达0–1 | 无轨迹→0并标注 | 速度曲线PNG |
| avg_acceleration | m/s² | 回合/选手 | 雷达0–1 | 同上 | 加速度曲线PNG |
| advance_ratio | 0–1 | 回合/选手 | 原值 | 三态失败→0 | 状态叠加图 |
| pause_ratio | 0–1 | 回合/选手 | 雷达展示1-pause | 同上 | 状态叠加图 |
| first_step_init | s | 回合 | 1-归一（越小越好） | 无信号→NA | 事件标记 |
| total_arm_extensions | 次 | 回合/选手 | 按阈归一 | 不可见→0并标注 | 区间标记 |
| launch_success_rate | 0–1 | 选手汇总 | 原值 | 无回合→0 | 冲刺标记 |
| attacking_ratio | 0–1 | 回合/选手 | 原值 | 判读失败→0 | 区间标记 |

### 16. 功能与模块详解

#### 16.1 两种报告/视图模式
- 经典视图（results.html）
  - 位置：`/results/<upload_id>`
  - 内容：
    - 下载初始 CSV（左/右 x/y 轨迹、meta）
    - 回合卡片：视频预览、标签（左/右）、AI 回合分析文本、AI 判罚（winner/置信度/理由）
    - 击剑手分析：关联左右击剑手；一键跳转 AI 深度分析；跨回合文本与下载
    - 整体性能对比：柱状图与雷达图（含图下分析文案）
    - 8 个战术图（左右各一）：攻击类型、节奏类型、攻击距离、反击机会、退却质量、退却距离、防守质量、回合结果模式；每图下方生成对应“战术分析”文案
    - 重新生成标签：`POST /reanalyze_video`
- 视频类型分析视图（video_view.html）
  - 位置：`/video_view/<upload_id>`
  - 内容：雷达图、触剑类型统计、AI 生成的胜负分析、Gemini 视频深度分析状态、图表轮播与即时战术要点。

#### 16.2 聊天/AI 教练
- 视频会话：`/chat/<upload_id>`（chat.html）
  - 行为：POST JSON `{ user_input }`，返回 `{ response }`；UI 支持打字指示、键盘回车发送、滚动保留历史；
  - 场景：询问某回合表现、请求训练建议、解释图表差异；
- 选手会话（Holistic Chat）：`/holistic_chat?fencer_id=...&weapon=...`
  - 结合选手汇总与武器类型，生成个体化分析与训练建议；
  - 依赖：`HolisticAnalysis` 记录与 `cross_bout_analysis_path`。

#### 16.3 标签（Tag）系统
- 数据结构：`Tag`、`BoutTag(bout_id, tag_id, fencer_side)`；
- 展示：经典视图每回合显示左右侧标签（TAG_TRANSLATIONS 中文映射）；
- 生成：
  - 自动：分析阶段抽取（见 `your_scripts/tagging.py` 调用路径于 `app.py::extract_tags_from_uploads`）；
  - 人工：后续可扩展 CRUD 接口（4A 已列路由）；
- 重新生成：`POST /reanalyze_video` 清理既有标签并重算；
- 用途：筛选复盘清单、在报告视图汇总热门标签并显示计数。

#### 16.4 选人（关联 Fencer）
- UI：经典视图"关联击剑手"表单（左右下拉）
- 数据：`Upload.left_fencer_id/right_fencer_id`；
- 作用：
  - 映射"回合胜负"到具体选手，统计按选手的胜/负/跳过；
  - 驱动选手级"Holistic Analysis"和 AI 会话（holistic_chat）。

#### 16.5 选 Winner 与 AI 判罚
- 选 Winner：`/select_bout_winners?upload_id=...`（select_bout_winners.html）
  - 逐回合展示视频与下拉选择 `left|right|skip`，提交后写回 `Bout.result`；
- AI 判罚（经典视图"AI判断"卡片）
  - 字段：`winner|confidence|reasoning`，显示于每回合卡片；
- 报告融合：report 视图的"各回合摘要表"标注 `result_source = ai|user`，以徽章区分。

#### 16.6 图表清单与产物（文件命名与文案）
- 总体对比图：
  - `fencer_analysis/plots/fencer_comparison.png`（柱状），图下显示 `chart_analysis.comparison_chart` 文案
  - `fencer_analysis/plots/fencer_radar_comparison.png`（雷达），图下显示 `chart_analysis.radar_chart` 文案
- 8 个战术图（左右各一）：目录 `fencer_analysis/advanced_plots/Fencer_Left|Right/`
  - 文件命名：`left_attack_type_analysis.png`、`left_tempo_type_analysis.png`、`left_attack_distance_analysis.png`、`left_counter_opportunities.png`、`left_retreat_quality.png`、`left_retreat_distance.png`、`left_defensive_quality.png`、`left_bout_outcome.png`（右侧同理 `right_*.png`）
  - 文案：`graph_analysis[key]` 渲染到每图下"战术分析"段落（无则提示"正在生成或暂不可用"）
- 选手档案图：`fencer_{id}_radar_profile.png`、`fencer_{id}_profile_analysis.png`（由 `create_fencer_radar_chart/create_fencer_analysis_chart` 生成）
- 其他：报告视图内各小图 `report.graphs.*` 与 `report.sections.*` 绑定（见 report.html）。

#### 16.7 Holistic Analysis（跨回合综合分析）
- 产物：`cross_bout_analysis.json` 与对应 `.txt` 文本；
- 入口：经典视图"下载分析报告"，并提供"左/右击剑手分析""视频聊天"按钮；
- 内容：
  - 汇总 `calculate_aggregated_metrics` 输出的选手级统计（速度/加速度/前进/暂停/首步/伸臂/冲刺成功/攻击比率）；
  - 汇总标签特征（`extract_tags_from_uploads`）；
  - 结合多视频来源的胜场/回合数对比条形图；
- 用途：训练处方生成、赛季纵向对比、课堂/汇报材料。

#### 16.8 三种武器（佩剑 saber / 花剑 foil / 重剑 epee）
- 字段：`Upload.weapon_type in {saber, foil, epee}`，在页面徽章显示；
- 参数差异：
  - 三态阈值与最小时长可按武器微调（见 6B 可覆盖项）；
  - 右路权/触剑事件窗口可独立设定（如花剑更关注权利判读，佩剑更关注节奏/速度突击）；
- 报告差异：报告头部标注武器类型，建议文案可按武器模板选择语气与重点。

### 17. 训练处方模板与判罚口径说明

#### 17.1 训练处方生成规则（基于八维雷达）
- 输入：选手汇总指标（速度、加速度、前进、暂停、首步、伸臂、冲刺成功、攻击比率），标签分布与回合摘要。
- 规则：
  - 低于阈值的维度进入"优先项"，设定具体目标（如：首步滞后从0.35s 降至 ≤0.25s）。
  - 结合标签负项（如 excessive_pausing、poor_attack_distance）映射到针对性练习。
  - 武器差异：
    - 佩剑：节奏与速度权重更高；短时高强度间歇；
    - 花剑：右路权判读与伸臂质量；
    - 重剑：距离管理与防守反击。
- 输出：周计划（频次×组数×时长）+ 测评口径（本周对比上周）。

#### 17.2 处方模板（示例）
- 模板A（首步与暂停控制）
  - 周目标：首步滞后下降≥20%，暂停比率下降≥10%。
  - 训练：
    - 反应起动（5×6×10s）：教练声光信号→首步爆发，计时记录；
    - 节奏压缩（4×5×20s）：3秒前进/1秒停顿循环，强调停顿缩短与重启速度；
  - 测评：本周最后一课复测首步滞后、暂停比率；
- 模板B（攻击距离与伸臂）
  - 周目标：良好攻击距离命中率+20%，过度伸臂标签-50%。
  - 训练：
    - 距离标尺（5×4×20s）：在地面标尺区间内保持最佳攻击距离；
    - 伸臂节拍（4×5×15s）：伸臂—击—收臂节拍器训练；
  - 测评：选取3回合复盘，记录"good_attack_distance/over_extension"标签变化。

#### 17.3 判罚口径说明（可解释）
- 右路权（以花剑为例）
  - 基本要素：先伸臂、先发起、先威胁；
  - 系统辅助：在回合摘要标注"先伸臂/先发起/节奏中断方"，并给出证据帧号；
  - 解释示例：
    - "第7回合左侧先伸臂并保持威胁，右侧中途暂停，AI判定左侧保有右路权（证据：t=1.23s 伸臂阈值达成，t=1.40s 右侧暂停）"。
- 节奏（tempo）
  - steady/variable/broken 的定义与判定来源于 advance 片段数量与变化；
  - 解释示例："第5回合右侧 broken tempo，三段前进被两次停顿打断，导致冲刺失败"。
- 距离（distance）
  - good/poor 距离口径：平均攻击距离接近最优阈（约2.0m±0.3m）；
  - 危险接近：连续帧数低于1.5m 计数>阈值；
  - 解释示例："第3回合左侧多次危险接近（<1.5m 共18帧），被右侧反击命中"。

#### 17.4 判罚与建议的落地
- 报告视图：在"各回合摘要"中区分 AI 与用户来源，异常回合加入"待复核"。
- 处方联动：从标签与指标短板自动勾选模板条目，填充为周计划草案，教练可编辑再下发。

### 18. 样例数据包与演示流程

#### 18.1 样例数据包（可用于评审演示）
- 目录：`results/1/159/`
  - `fencer_analysis/`：
    - `fencer_Fencer_Left_analysis.json`、`fencer_Fencer_Right_analysis.json`
    - `advanced_plots/Fencer_Left/*.png`、`advanced_plots/Fencer_Right/*.png`
  - `match_analysis/`：`*_analysis.json`（回合级）
  - `csv/`：`left_xdata.csv`、`left_ydata.csv`、`right_xdata.csv`、`right_ydata.csv`、`meta.csv`
  - `matches/`：回合剪辑（如已生成）

#### 18.2 演示流程（10–15分钟）
1) 登录并进入上传页，选择样例视频（或复用已处理的 upload_id=159）；
2) 打开经典视图：滚动查看回合卡片、标签、AI 分析与判罚；
3) 切换报告视图：展示执行摘要、KPI、小图集与各回合摘要表；
4) 展示 8 个战术图与图下"战术分析"文案；
5) 展示"选择回合获胜者"页面并说明 AI 与用户来源如何在报告内区分；
6) 打开聊天页，询问一个具体回合的提升建议；
7) 下载 CSV/JSON/PNG 并展示本地文件结构；
8) 如有时间，演示"关联击剑手"并进入 Holistic Chat。

#### 18.3 评审要点
- 页面跳转路径清晰、产物可下载、报告可打印；
- 图表命名与口径一致，中文字体正确；
- 回合数、结果来源、标签分布与建议逻辑一致。

### 19. 一键导出 PDF/Word 说明

#### 19.1 PDF 导出
- 报告视图 `report.html` 支持浏览器"打印为PDF"；
- 建议：Chrome/Edge 打印并选择"背景图形"和"高质量"；页眉页脚可由登记系统自动生成。

#### 19.2 Word/Docx 导出（方案）
- 方式A：将 `report` 数据对象喂给模板引擎（如 Jinja→HTML），再用 `docx` 转换脚本生成 `.docx`；
- 方式B：直接用 `python-docx` 读取 `report` 数据，构建章节、表格与图片插入；
- 命名：`report_{upload_id}_{timestamp}.docx`；
- 注意：确保图表以 300DPI 导出并按 `results/...` 路径读取插入。

#### 19.3 批量导出（可选）
- 输入：上传ID列表或时间范围；
- 过程：遍历生成 PDF/Docx，并打包 ZIP；
- 权限：仅限本人数据；
- 审计：记录导出时间、范围与产物清单。

### 20. 报告体系深度说明

#### 20.1 模式与适用场景
- 经典视图：适合逐回合教学复盘与精细标注；
- 报告视图：适合家校沟通/俱乐部周报/打印归档；
- 触剑类型分析：适合战术专题教学与裁判培训。

#### 20.2 组成与版式
- 执行摘要：分数、胜负、亮点；
- KPI 面板：速度/加速度/前进/暂停/首步/冲刺成功等左右对比；
- 总体图表：柱状与雷达，并配"左右要点" bullets；
- 战术小图集：进攻与防守分栏，左右各图 + bullets；
- 各回合摘要表：结果来源徽章（AI/用户）、关键指标列；
- 标签与建议：热门标签 TopN 与侧别建议；
- 附录与下载：CSV/JSON 列表。

#### 20.3 生成流水线
1) 收集回合级 metrics 与标签；
2) 聚合选手级统计与胜负；
3) 渲染图表（确保中文字体与DPI）；
4) 产出 `report` 数据对象（见 report.html 的 `report` JSON）；
5) 模板填充渲染页面，支持打印为PDF；
6) 产物写入 `results/{user}/{upload}` 并登记到下载清单。

#### 20.4 自定义与重构点
- 主题/配色/字体：可在 CSS 层修改；
- 指标列：可增删 KPI 或增加新图；
- 版式：可切换单双栏、分章节打印；
- 文案：`chart_analysis` 与 `graph_analysis` 可由模型生成或人工编辑；
- 多视频合并：通过 `UploadVideo` 与 `bouts_offset` 撑起整场赛事报告。

#### 20.5 验收与抗错
- 图表完备性（空图占位提示）、字段一致性、中文渲染、下载可用；
- 失败图表记录到日志并可"重渲染"；
- 大图量使用分批写盘避免IO拥塞。

### 21. 分析图表深度说明

#### 21.1 总体图表（柱状、雷达）
- 输入：选手级 metrics；
- 算法：见 6A/6B；
- 解读：柱状侧重绝对差异；雷达侧重维度轮廓与短板定位；
- 易错：全部为0时提供占位与"上传视频以生成数据"的提示。

#### 21.2 八个战术图（左右各一）
- 攻击类型与胜利（attack_type_analysis）
  - 输入：advance_analyses.attack_info；
  - 含义：不同攻击类型（simple/compound/holding/preparation）与胜率；
  - 易错：无攻击回合→应显示"no_attacks"。
- 节奏类型（tempo_type_analysis）
  - 输入：advance_analyses.tempo_type（steady/variable/broken）；
  - 含义：节奏稳定性与胜负的关系；
- 攻击距离（attack_distance_analysis）
  - 输入：advance_analyses.avg_distance/min_distance；
  - 含义：距离与命中效率；
- 反击机会（counter_opportunities）
  - 输入：retreat_analyses.opportunities_taken/missed；
  - 含义：退中是否出现有效反击窗口；
- 退却质量（retreat_quality）
  - 输入：distance_management_quality、spacing_consistency；
  - 含义：安全距离维持与间距稳定；
- 退却距离（retreat_distance）
  - 输入：retreat间隔的平均距离；
  - 含义：拉开距离的能力；
- 防守质量（defensive_quality）
  - 输入：reaction_type、composure、reaction_quality；
  - 含义：防守姿态与质量；
- 回合结果模式（bout_outcome）
  - 输入：winner_side 与上述要素聚合；
  - 含义：哪些策略与模式指向胜利。

- 每图产物：`advanced_plots/Fencer_Left|Right/<side>_<key>.png`；
- 每图文案：`graph_analysis[key]` 注入页面"战术分析"。

#### 21.3 图表一致性与校验
- 维度中文名与英文 key 对齐；
- 图例与颜色语义一致（左=绿、右=红）；
- 同一维度跨视图口径一致（见6A）。

### 22. 聊天系统深度说明

#### 22.1 Prompt 与上下文
- 上下文：`upload_id`、回合摘要、标签、关键指标与图表路径；
- Prompt 模板：围绕"提问理解→引用证据→生成建议"；
- 安全：限制外部链接与隐私信息；
- 语言：默认中文，支持英文问答（双语能力取决于模型）。

#### 22.2 记忆与状态
- 会话上下文存于服务器会话（或数据库）；
- 支持对同一上传的多轮追问；
- 计划：关联 `Fencer` 建立选手级会话历史（可选）。

#### 22.3 典型问法与效果
- "我的首步为什么慢？请基于第3、7、9回合给建议。"
- "对比左右雷达，列3条训练重点，配练习内容与次数。"
- "本场进攻距离管理是否到位？证据帧与回合。"

#### 22.4 容错与退出
- 网络或模型失败→回退为固定模板建议；
- 遇到空数据→引导先完成视频分析；
- 超长输入→截断并提示。

#### 20A. 报告可配置清单（开关/顺序/样式/颗粒度）
- 开关：执行摘要、KPI、总体图、战术小图集、各回合表、热门标签、建议、附录下载；
- 顺序：模块可重排；
- 样式：主题色、中文字体、卡片阴影、单双栏布局；
- 颗粒度：
  - KPI 列：可增删（如加入"伸臂时长/次数"）；
  - 小图集：可按键选择（attack_distance/retreat_quality等）；
  - 表格列：可选显示"first_step/acceleration"等；
- 出口：PDF/Docx 批量导出、ZIP 打包；
- 模板：校园/俱乐部/赛事三套预设（标题、口吻、重点）

#### 21A. 图表参数与阈值对照表（与6B衔接）

| 图表 | 关键参数 | 默认值 | 说明 |
|---|---|---|---|
| 柱状对比 | dpi,font | 300, SimHei/DejaVu | 打印清晰度 |
| 雷达图 | radar_scale | 0–1 | 归一化范围 |
| 攻击类型 | has_attack阈值 | 基于attack_info | 判定是否计入 |
| 节奏类型 | tempo变化阈值 | ≤2 为 variable | 大于2 为 broken |
| 攻击距离 | optimal±band | 2.0±0.3m | 可按武器覆盖 |
| 反击机会 | opp_taken/missed | 合计≥1 | 小样本高亮 |
| 退却质量 | spacing_variance阈值 | 经验阈 | 一致性评估 |
| 退却距离 | avg_retreat_dist | 经验阈 | 拉开能力 |
| 防守质量 | reaction_quality枚举 | good/avg/poor | 统一口径 |
| 回合结果 | winner_side | left/right/skip | 源自AI或用户 |

说明：阈值可按武器（saber/foil/epee）与数据集标定调整。

#### 22A. 聊天模板库（按身份/场景）
- 教练：
  - "请基于第{idxs}回合，给出3条训练处方（频次×组数×时长），目标：{指标} 从 {当前值} 提升到 {目标值}。"
  - "对比左右的雷达短板，列出本周课堂的示范案例主题与视频片段帧号。"
- 运动员：
  - "我的冲刺老是失败，结合第{idx}回合的速度/加速度曲线，指出时机和距离问题，并给2个居家练习。"
  - "请把我这次比赛的优势写成3条积极反馈，便于家长沟通。"
- 裁判/技术：
  - "第{idx}回合是否满足右路权？请引用伸臂/发起/暂停证据帧。"
  - "本场 broken tempo 的典型案例有哪些？给出回合号与简要原因。"
- 系统与容错：
  - 无数据提示："当前上传尚未完成分析，请先在结果页确认状态或重新触发分析。"
  - 过长输入："为保证响应速度，系统已截断，请聚焦1–3个回合或1–2个指标。"

### 1A. 附录与扩展章节索引（主文摘要与跳转）

| 附录/扩展章节 | 主文摘要 | 主文引用位置 |
|---|---|---|
| 附加A · 2A表格版（含基线指标） | 将第2节"需求分解"以表格呈现，给出每条需求的验收口径/度量/数据来源/备注与基线 | 第2节末"2A摘要"与第1节概览；详见"附加A" |
| 附加B · 6A参数默认值与可配置项 | 汇总算法与指标的默认参数与可配置项（按武器可覆盖） | 第6节"参数配置（6B）"与"6F参考表"指向；详见"附加B" |
| 附加C · 9C运维仪表盘与备份恢复模板 | 定义Prom指标、阈值与SOP（备份/恢复/演练模板） | 第9节"运维与安全细化（9A/9B）"摘要与索引；详见"附加C" |
| 附加D · 用户价值详解 | 分角色价值、场景流程、ROI与采纳路线的完整展开 | 第2节已纳入"摘要版"，详见"附加D"获取细节 |
| 3X 实施细则 | 3.2–3.11 每一步的操作/API/参数/监控/验收/故障 | 第3节末"实施细则"摘要与跳转 |
| 4A–4D 接口/数据字典/示例/错误码 | 全量API清单、JSON/CSV字典、请求/响应示例、错误码规范 | 第4节"索引：4A/4B/4C/4D"摘要与跳转 |
| 5G–5P 数据框架 | 数据流转、存储布局、质量校验、契约与版本、导入导出、权限、血缘、留存、恢复、端到端样例 | 第5节末"框架扩展（5G–5P）"摘要与跳转 |
| 6A–6F 算法与指标 | 正式口径表、公式、参数、误差与不确定性、图证、验收 | 第6节"口径与验证"摘要与跳转 |
| 16 功能与模块详解 | 三种报告模式、聊天、标签、选人、选winner与AI判罚、图表产物、Holistic Analysis、三种武器 | 第1节与第4节中概述并指向16节细化 |
| 17 训练处方与判罚口径 | 基于雷达短板的处方模板与右路权/节奏/距离的可解释判罚话术 | 第6节与第16节"建议与判罚"摘要与跳转 |
| 18–19 样例与导出 | 演示脚本与数据包、PDF/Docx导出与批量打包 | 第3节与第20节末"演示与导出"摘要与跳转 |
| 20–22 报告/图表/聊天深度说明 | 报告流水线与自定义、八图算法与解读、聊天上下文与模板 | 第4节、6节、16节中各自摘要与索引 |
| 20A 报告可配置清单 | 模块开关、顺序、样式、颗粒度、三套模板 | 第20节末"20A摘要" |
| 21A 图表参数与阈值表 | 各图关键阈值默认值与说明 | 第21节末"21A摘要" |
| 22A 聊天模板库 | 教练/运动员/裁判场景模板与容错提示 | 第22节末"22A摘要" |

说明：主文各节均在末尾增加"索引/跳转"提示，以便快速定位附录与扩展章节的详细内容。

#### 7.1 信息架构与导航
- 导航：上传→状态→结果（经典/报告/触剑分析）→聊天/Holistic；
- 面包屑与返回按钮统一；
- 关键操作固定在顶部按钮区（经典/报告切换、选择winner、聊天）。

#### 7.2 结果页布局（可读性）
- 回合卡片分组展示：视频、标签、分析、判罚；
- 图表采用卡片+标题+说明文字（避免纯图无解释）；
- 颜色语义：左=绿、右=红、一致贯通。

#### 7.3 空态与异常态
- 无数据：卡片内给出生成指引；
- 渲染失败：错误提示+重渲染建议；
- 权限不足：引导登录或联系管理员。

#### 7.4 可用性与可访问性
- 按钮触达范围大、键盘可操作、图片alt文本；
- 颜色对比度符合阅读要求；
- 移动端栅格自适应、视频自适应。

#### 7.5 交互细节
- Loading/打字指示、滚动定位、锚点链接（跳至某回合/某图表）；
- 下载网格与状态提示；
- 标签hover 展示中文翻译（TAG_TRANSLATIONS）。

#### 7.6 文案与术语
- 指标中文名与英文 key 保持对照（见6A/21A）；
- 判罚/建议采用"证据+建议"结构，减少模糊词；
- 统一使用"回合、节奏、伸臂、冲刺"等规范词。

#### 7.7 可配置UX项
- 卡片阴影、主题色、字号、单双栏；
- KPI顺序、图表可见性；
- 导出时是否包含建议段落。

#### 7.8 任务与进度反馈
- 状态页显示队列进度与阶段（检测/跟踪/切分/渲染）；
- 失败提供"查看日志""重试"按钮；
- 历史任务列表可按时间/状态过滤。

#### 7.9 UX度量
- 页面加载时间、图表渲染时间、点击/下载转化；
- 聊天留存、对话轮数、积极反馈率；
- 任务完成率与复盘完成率。

#### 7.10 索引
- 参见：20 报告、21 图表、22 聊天、18 演示。

---

#### 8.1 身份认证与会话
- Flask-Login；会话过期、强制登出；
- 异步接口校验登录态；
- CSRF 防护开启。

#### 8.2 访问控制
- 仅允许访问本人 `results` 子树；
- 下载白名单与目录穿越防御；
- 管理员/审计最小化权限（见5L）。

#### 8.3 数据最小化与脱敏
- 日志脱敏（路径、账号、密钥）；
- 导出不含隐私字段；
- 错误响应不暴露堆栈细节于用户端。

#### 8.4 传输与存储安全
- HTTPS/TLS（Nginx反代）；
- 密钥配置走环境变量；
- 备份加密与密钥轮换（9A/9C）。

#### 8.5 依赖与镜像安全
- 依赖扫描（SAST/DAST）、SBOM与镜像签名；
- 固定版本、最小攻击面。

#### 8.6 合规与授权
- 素材来源与授权提示；
- 未成年人/肖像权/著作权遵循；
- 数据留存与擦除策略（5N）。

#### 8.7 审计与追踪
- 下载与共享记录；
- trace-id 贯穿链路；
- 生成报告保留 `schema_version/params_hash`。

#### 8.8 安全告警
- 登录失败频繁、下载异常、5xx 尖刺；
- 阀值见 9A；
- 通知渠道：邮件/IM。

#### 8.9 渗透与越权用例
- 下载目录穿越、路径注入、跨用户访问；
- 结果：应全部阻断并记录；
- 定期回归。

#### 8.10 索引
- 参见：9 运维、5 框架、4D 错误码。

---

#### 9.1 部署拓扑
- Web(Flask) + Worker(Celery) + Redis + DB + FFmpeg + Nginx；
- GPU 节点优先；
- 目录：`Uploads/`、`results/` 可写。

#### 9.2 配置与环境
- `.env`：密钥/DB/Redis/模型路径；
- `requirements.txt`：锁定版本；
- 字体：中文字体包。

#### 9.3 启停与自愈
- Supervisor/systemd 管理进程；
- Worker 掉线自动拉起；
- 队列堵塞自动告警。

#### 9.4 监控看板
- 资源：CPU/MEM/DISK/GPU；
- 应用：QPS、P95、失败率、队列长度；
- 产物：目录容量、图表生成量（见9A）。

#### 9.5 告警与SLA
- P1：不可用/数据风险 15min；P2：性能退化 2h；
- 阈值参考 9A；
- 值守排班与应急通讯录。

#### 9.6 备份恢复
- 周全量、日增量；
- 恢复演练季度进行，记录 RTO/RPO；
- 校验哈希与可读性（附加C模板）。

#### 9.7 变更与发布
- 预生产验证→灰度→全量→回滚脚本；
- 配置变更审计。

#### 9.8 容器与编排（可选）
- 镜像多阶段构建；
- GPU 插件、HPA、Ingress/TLS；
- 命名空间隔离与网络策略。

#### 9.9 容量与成本
- 每上传产物规模预估；
- 存储分层（热/冷）；
- 带宽与GPU时长测算（可附表）。

#### 9.10 运维SOP
- 故障等级、止血、根因五问、复盘闭环；
- 工单与知识库；
- 周/月报模版。

#### 9.11 合规与审计
- 导出审计、留存周期、擦除流程；
- 第三方合规检查清单。

#### 9.12 索引
- 参见：9A/9B/9C 扩展、19 导出。

---

#### 10.1 性能目标
- 单视频分析：GPU 环境接近实时；CPU 环境可用但较慢；
- 报告渲染成功率≥99%，P95渲染耗时受控。

#### 10.2 稳定性目标
- 任务成功率≥99%；
- 切分 F1≥0.9；
- 失败可重试并有回退。

#### 10.3 关键优化点
- 检测与跟踪：阈值与轻量权重、分段缓存；
- IO：顺序写、批处理；
- 图表：并发渲染、降重绘。

#### 10.4 资源与成本测算
- GPU 显存/时长、CPU 核心、内存、磁盘；
- 不同规模（校园/俱乐部/赛事）三档建议。

#### 10.5 压测与基准
- 样例视频集、并发上传、长视频；
- 指标：QPS、P95、失败率、显存峰值。

#### 10.6 降级与回退
- 无GPU→CPU 路径；
- 大图量→延迟渲染/批量导出；
- 转码失败→保留原视频。

#### 10.7 风险与缓解
- 模型与驱动不兼容→容器封装；
- 目录爆满→清理与配额；
- 漏洞→依赖升级与扫描。

#### 10.8 指标看板
- 任务成功率、切分F1、渲染成功率、复盘完成率；
- 资源：GPU/CPU/DISK；
- 体验：加载时间、聊天留存。

#### 10.9 成本优化
- 压缩/降分辨率、批处理、冷热分层；
- 模型蒸馏与轻量化。

#### 10.10 索引
- 参见：6A–6F 算法、9 运维、21 图表。

