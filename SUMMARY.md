## 项目概览（简要）

- **定位**: 击剑比赛（军刀）数据分析与可视化。包含对回合的运动学分析、触发类型判别、攻防专项指标、以及选手画像图谱与整合报告。
- **主要目录**:
  - `your_scripts/`: 分析、分类、可视化与画像核心代码。
  - `templates/report.html`: 中文报告模板与前端展示样式。
  - `results/<upload_id>/...`: 分析输出（单回合与跨回合统计、图像）。

## 关键流程

1) 原始回合数据 → `your_scripts/bout_analysis.py`
- 核心函数：`analyze_fencing_bout(...)`、`process_match(...)` 等，完成：
  - 位置特征提取、速度/加速度计算与平滑
  - 进/退/停区间与节奏识别、臂伸/起跳（lunge/launch）检测
  - 进攻类型与间距、首步时机等战术要素摘要

2) 回合类型判别 → `your_scripts/bout_classification.py`
- `classify_fencer_touch_category(...)`：基于区间占比与末段动作，判定 `in_box | attack | defense`。
- `classify_bout_touches(...)`：双侧回合分类并融合胜负信息。
- `aggregate_touch_statistics(...)`：聚合胜率与类别数量统计。

3) 专题可视化与指标 → `your_scripts/touch_visualization.py`
- 生成类别分布/胜率对比、In-Box、速度-加速度散点、首步时机等图表。
- 含多项衍生指标计算（如主动率、臂伸率、时机精度、攻防效率等）。

4) 攻防深度分析 →
- 进攻：`your_scripts/attack_comprehensive_analysis.py`
  - `extract_attack_bout_details(...)`：提取攻方速度、加速度、攻距、臂伸/起跳、节奏类型等。
  - `create_comprehensive_attack_charts(...)`：攻距趋势、类型/节奏胜负柱状、KPI 仪表盘等。
- 防守：`your_scripts/defense_comprehensive_analysis.py`
  - `extract_defense_bout_details(...)`：退防距离、安全距离维持、一致性、反击机会与利用率等。
  - `create_comprehensive_defense_charts(...)`：防守距离、反击利用、退防一致性、KPI 等。

5) 选手画像整合 → `your_scripts/fencer_profile_plotting.py`
- 提供画像雷达、攻型分布、节奏/距离/过程图、胜因分析等图表。
- `generate_fencer_profile_graphs(upload_id, base_output_dir)` 产出画像图集。

6) 画像与分析落地 → `your_scripts/fencer_profile_integration.py`
- `update_fencer_analysis_with_enhanced_tags(upload_id, ...)`：基于回合数据生成并写回增强标签与统计。
- `integrate_fencer_profile_graphs_with_analysis(upload_id, ...)`：按左右选手生成画像图并注册到 `cross_bout_analysis.json`。
- `generate_comprehensive_fencer_profile_report(upload_id, ...)`：编排完整“标签+画像+摘要”报告字典。

7) 图表解读（可选）→ `your_scripts/graph_analysis.py`
- 使用 Gemini 生成面向教练的中文精炼解读（攻型/节奏/距离/反击/退防质量等）。

## 主要函数速览（摘）

- `bout_analysis.py`: `analyze_fencing_bout`, `detect_launches_in_advance_intervals`, `detect_arm_extensions_in_advance_intervals`, `analyze_all_intervals`
- `bout_classification.py`: `classify_fencer_touch_category`, `classify_bout_touches`, `aggregate_touch_statistics`
- `touch_visualization.py`: `create_touch_category_charts`, `create_inbox_analysis_charts`, 以及多项指标计算函数
- `attack_comprehensive_analysis.py`: `extract_attack_bout_details`, `create_comprehensive_attack_charts`
- `defense_comprehensive_analysis.py`: `extract_defense_bout_details`, `create_comprehensive_defense_charts`
- `fencer_profile_plotting.py`: `save_fencer_profile_plots`, `generate_fencer_profile_graphs`
- `fencer_profile_integration.py`: `update_fencer_analysis_with_enhanced_tags`, `integrate_fencer_profile_graphs_with_analysis`, `generate_comprehensive_fencer_profile_report`
- `graph_analysis.py`: `analyze_attack_type_graph`, `analyze_tempo_type_graph`, `analyze_attack_distance_graph`, `analyze_counter_opportunities_graph`, `analyze_retreat_*`

## 输出与报告

- 结构：`results/<upload_id>/`
  - `match_analysis/`: 单回合 JSON 与图
  - `fencer_analysis/`: 跨回合统计、`profile_plots/` 画像图
- 前端模板：`templates/report.html`（中文 UI、KPI 组件、样式主题）。

## 运行与依赖（简）

- 依赖：见根目录与子模块 `requirements.txt`；含 `matplotlib/numpy` 等科学计算与绘图库。
- 交互/可视化：`templates/report.html`；视频追踪与修复相关代码位于 `your_scripts/tracker/` 与 `your_scripts/inpainter/`（如 `XMem`、`E2FGVI`）。

