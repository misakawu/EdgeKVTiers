# h1/README.md

## 概述

`h1/` 目录包含 H1 阶段的核心实验框架，基于 **vLLM 0.11.0** 的 GPU 前缀缓存（Prefix Caching）实现，用于比较四种缓存策略在不同 GPU 内存预算下的表现：

| 策略名 | 说明 |
|--------|------|
| `vllm_default` | vLLM 原生 LRU（默认行为，对照基线） |
| `h1_lru` | 自定义 LRU（与原生等价，用于诊断对照） |
| `h1_lfu` | 基于访问频率的 LFU 策略 |
| `h1_lpe` | 基于对象级 LPE 分数（`p_reuse * c_recomp / size`）的策略 |

**核心机制**：通过 `sitecustomize.py` 在运行时钩入（monkey-patch）vLLM 的 `BlockPool`，接管 GPU 缓存块的分配、驱逐和重排逻辑，同时收集细粒度的 GPU 缓存统计和对象级画像。

---

## 核心实验脚本

### `run_h1_vllm0110_real.py` — 核心实验引擎

真正的实验执行入口。在 offline vLLM 环境（`LLM.generate`）中直接运行 replay trace，支持多策略、多预算、可配置并发和批处理顺序。输出请求级 CSV、summary JSON 和 GPU 监控数据。

#### 关键环境变量（控制策略和运行时行为）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `EDGEKV_H1_GPU_POLICY` | 策略名：`vllm_default` / `h1_lru` / `h1_lfu` / `h1_lpe` | `vllm_default` |
| `EDGEKV_H1_PROFILE_POLICY_TIME` | 是否统计策略决策耗时（微秒） | `1` |
| `EDGEKV_H1_STATS_DIR` | GPU 统计 JSON 输出目录 | `out_dir/edgekv_gpu_stats` |
| `EDGEKV_H1_RUNTIME_MONITOR` | 是否启用 LPE 运行时 JSONL 监控日志 | `0`（LPE 策略自动开启） |
| `EDGEKV_C_RE_MS_PER_TOKEN` | 每 token 重计算耗时（ms） | `0.12` |
| `EDGEKV_MU_KV_MB_PER_TOKEN` | 每 token KV 缓存大小（MB，用于理论回退） | 由模型自动推导 |

#### 主要命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out` | 输出目录 | `h1/out/h1_vllm_real` |
| `--model` | 模型路径（支持本地或 HuggingFace） | `models/Qwen2.5-7B-Instruct` |
| `--dtype` | 数据类型 | `float16` |
| `--policies` | 策略列表（空格分隔） | `vllm_default h1_lru h1_lfu h1_lpe` |
| `--budgets` | 预算名（`tight`/`mid`/`loose`）或数值（如 `0.75`） | `tight mid loose` |
| `--replay-trace` | replay trace JSONL 路径 | `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` |
| `--max-requests` | 最大请求数 | `1024` |
| `--replay-batch-size` | 每批请求数（并发度） | `1` |
| `--batch-order` | 批处理顺序（`original` / `length_bucket`） | `original` |
| `--warmup-batches` | 预热批次数（不计入测量） | `0` |
| `--tensor-parallel-size` | TP 大小 | `2` |
| `--max-model-len` | 最大序列长度 | `2048` |
| `--visible-devices` | 可见 GPU 设备 | `0,1` |

**用法示例**：

```bash
# 单个策略 × 单个预算
python h1/run_h1_vllm0110_real.py --policies h1_lpe --budgets 0.80 --replay-batch-size 16 --out ./my_exp

# 完整矩阵（4 策略 × 3 预算）
python h1/run_h1_vllm0110_real.py --policies vllm_default h1_lru h1_lfu h1_lpe --budgets tight mid loose
```

---

### `run_step3_budget_tiers.py` — 单 Tier 实验编排

运行一个 tier（如 `tight`）下的所有 `预算 × 策略` 组合，底层调用 `run_h1_vllm0110_real.py`。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visible-devices` | GPU 设备 | `0,1` |
| `--tier` | 实验标签（如 `tight`） | `tight` |
| `--base-out` | 基础输出目录 | `h1/out/step3` |
| `--budgets` | 预算列表（空格分隔） | `tight mid loose` |
| `--policies` | 策略列表 | `vllm_default h1_lru h1_lfu h1_lpe` |
| `--num-prompts` | 请求数 | `1024` |
| `--replay-trace` | trace 路径 | 默认 |
| `--replay-batch-size` | 批大小 | `16` |
| `--force` | 强制重跑 | 否 |
| `--keep-cells` | 保留中间文件 | 否 |
| `--no-finalize` | 跳过汇总和清理（用于多 rep 协议） | 否 |

**用法**：

```bash
python h1/run_step3_budget_tiers.py --tier my_tier --budgets 0.75 0.80 --policies h1_lpe
```

---

### `run_step3_repeat.py` — 可重复协议（多 Rep）

在预设工作点（`pressure_tight_n400`, `pressure_mid_n400`, `pressure_loose_n400`）上运行完整的 4 策略 × 3 预算矩阵，每个 cell 重复 `reps` 次（默认 3），最后汇总中位数并输出可视化。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visible-devices` | GPU | `0,1` |
| `--reps` | 重复次数 | `3` |
| `--force` | 重跑 | 否 |
| `--keep-cells` | 保留中间文件 | 否 |

**用法**：

```bash
python h1/run_step3_repeat.py --reps 5 --force
```

---

### `run_step3_real.py` — 真实混合 Trace 多 Rep 实验

与 `run_step3_repeat.py` 类似，但使用真实混合 trace（`sharegpt_hotpotqa_session.jsonl`），并支持**并发扫描**（`--batch-sweep`）以确定最佳并发度。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visible-devices` | GPU | `0,1` |
| `--reps` | 重复次数 | `3` |
| `--budgets` | 预算列表 | `tight mid loose` |
| `--policies` | 策略列表 | 全部 |
| `--max-requests` | 最大请求数 | `1024` |
| `--replay-batch-size` | 默认并发 | `32` |
| `--batch-sweep` | 并发扫描值列表（如 `8 16 32 64`） | 无 |
| `--recommended-batch-sweep` | 使用推荐值 `8 16 32 64` | 否 |
| `--batch-order` | 批顺序 | `original` |
| `--warmup-batches` | 预热批数 | `0` |
| `--force` | 重跑 | 否 |

**用法**：

```bash
# 并发扫描，确定最佳 batch size
python h1/run_step3_real.py --batch-sweep 8 16 32 64

# 只跑单一并发和部分预算/策略
python h1/run_step3_real.py --replay-batch-size 32 --budgets tight mid --policies h1_lpe
```

---

## 定档与扫描脚本

### `run_find_interval_2.py` — 粗网格定档扫描

在 `0.77~0.90` 区间用粗网格（默认 `0.77 0.80 0.83 0.86 0.88 0.90`）扫描 LRU 命中率，为三个目标命中率（`0.5, 0.7, 0.9`）选取最近点。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--coarse-budgets` | 扫描预算值 | `0.77 0.80 0.83 0.86 0.88 0.90` |
| `--policies` | 策略 | `h1_lru h1_lpe` |
| `--reference-policy` | 参考策略（用于定档） | `h1_lru` |
| `--target-hits` | 目标命中率 | `0.5 0.7 0.9` |
| `--force` | 重跑 | 否 |

**输出**：`find_interval_2_report.csv/json`，含三档最近点及 LPE 相对 LRU 的 p95 延迟增益。

**用法**：

```bash
python h1/run_find_interval_2.py --coarse-budgets 0.78 0.82 0.86 0.90
```

---

### `run_find_hit.py` — 候选取线生成与曲线判断

生成多组候选 trace（A-E 组），每组运行 LRU 预算扫描，判断曲线是否平滑，并自动归档 trace 和结果到备份目录。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--groups` | 组名列表 | `A B C D E` |
| `--budgets` | 预算列表 | `0.77 0.80 0.83 0.86 0.88 0.90` |
| `--policies` | 策略 | `h1_lru` |
| `--reference-policy` | 参考策略 | `h1_lru` |
| `--skip-trace-generation` | 跳过生成 trace | 否 |
| `--skip-run` | 跳过运行实验 | 否 |
| `--force` | 重跑 | 否 |
| `--overwrite-backups` | 覆盖备份 | 否 |

**用法**：

```bash
python h1/run_find_hit.py --groups E --policies h1_lru h1_lpe
```

---

## 汇总与可视化

### 汇总脚本（Summarizers）

这些脚本读取各次实验输出的 `*_summary.json`，按 `(预算, 策略)` 分组，计算中位数（或均值±标准差），生成简洁的 CSV 汇总文件。

| 脚本 | 适用场景 | 关键参数 |
|------|----------|----------|
| `summarize_step3_budget_tiers.py` | 单 tier 输出 | `--out`, `--summary` |
| `summarize_step3_real.py` | 多 rep 真实实验 | `--base`, `--summary` |
| `summarize_step3_repeat.py` | 多 rep 压力协议 | `--base`, `--summary` |

**用法示例**：

```bash
python h1/summarize_step3_real.py --base h1/out/step3_real --summary h1/out/step3_real/step3_real_summary.csv
```

---

### 可视化脚本

根据汇总 CSV 绘制 4 策略 × 3 预算的对比柱状图，保存为 PNG + PDF。

| 脚本 | 输出内容 |
|------|----------|
| `visualize_step3_real.py` | tight/mid/loose 三个预算的指标对比 |
| `visualize_lpe_scenarios.py` | A/B/C 三个压力场景的指标对比 |

**用法**：

```bash
python h1/visualize_step3_real.py --summary h1/out/step3_real/step3_real_summary.csv
```

---

## 验证工具

### `validate_d3.py` — D3 诊断验证

检查 LPE 策略的三个 D3 核心指标：

1. `c_recomp` 与 `n_tokens` 是否线性（`std(c_recomp/n_tokens) < 1e-5`）
2. 驱逐粒度是否为 `vllm_prefix_cache_block`
3. score 与 p_reuse 是否高度相关且 score 非零

| 参数 | 说明 |
|------|------|
| `stats_dir` | `edgekv_gpu_stats` 目录路径 |
| `--out-json` | 输出 JSON 路径（默认自动生成） |

**用法**：

```bash
python h1/validate_d3.py h1/out/step3_real/rep1/tight_h1_lpe/edgekv_gpu_stats
```

---

### `validate_d4_saturation.py` — D4 饱和度验证

基于 summary JSON 和请求级 CSV，估算实际吞吐量 vs 理论预填充吞吐量，判断是否进入队列饱和状态。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `summary_json` | 实验 summary JSON | 必填 |
| `requests_csv` | 请求级 CSV | 必填 |
| `--c-re-ms-per-token` | 每 token 重计算时间（ms） | 从 summary 读取 |
| `--threshold` | 饱和度阈值（实际/理论 < 阈值 则判定饱和） | `0.60` |
| `--out-json` | 输出路径 | 可选 |

**用法**：

```bash
python h1/validate_d4_saturation.py h1/out/.../summary.json h1/out/.../requests.csv --threshold 0.5
```

---

### `aggregate_h1_serving_bench.py` — 旧版聚合器

用于从旧的 `serving_bench` 输出（`result.json` + `stats/` 目录）聚合出 CSV。现已被 `run_h1_vllm0110_real.py` 直接输出取代，保留用于兼容旧数据。

| 参数 | 说明 |
|------|------|
| `--out` | 输出目录 |
| `--budgets` | 预算列表 |
| `--scrape-metrics` | Prometheus metrics 端点 URL |
| `--metrics-out` | metrics 输出文件 |

---

## 其他辅助文件

### `sitecustomize.py` — 运行时策略注入与统计收集（无 CLI）

此文件在 Python 启动时自动导入（通过 `PYTHONPATH` 机制），是整个 H1 实验的**核心运行时引擎**。它通过 monkey-patch 的方式钩入 vLLM 的 `BlockPool` 和 `KVCacheManager`，实现策略替换、统计收集和对象级 Profile 管理。

**关键环境变量**：

| 分类 | 环境变量 | 说明 | 默认值 |
|------|----------|------|--------|
| **策略选择** | `EDGEKV_H1_GPU_POLICY` | 策略名 | `vllm_default` |
| | `EDGEKV_H1_PROFILE_POLICY_TIME` | 是否记录策略决策耗时 | `1` |
| **LPE 核心** | `EDGEKV_C_RE_MS_PER_TOKEN` | 每 token 重计算耗时（ms） | `0.12` |
| | `EDGEKV_MU_KV_MB_PER_TOKEN` | 每 token KV 缓存大小（MB） | 模型自动推导 |
| **LPE 权重** | `H1_LPE_W_FREQ` | 频率权重 | `0.55` |
| | `H1_LPE_W_RECENCY` | 近因权重 | `0.30` |
| | `H1_LPE_W_TYPE` | 类型先验权重 | `0.15` |
| | `H1_LPE_W_PRIOR` | 外部先验权重 | `0.70` |
| **LPE 调度** | `H1_LPE_REORDER_MODE` | 重排模式 `window`/`full`/`off` | `window` |
| | `H1_LPE_REORDER_WINDOW` | 重排窗口大小 | `128` |
| | `H1_LPE_LIGHT_PATH` | 低压力时跳过重排（优化性能） | `1` |
| | `H1_LPE_PRESSURE_FREE_RATIO` | 触发重排的空闲块比例阈值 | `0.15` |
| | `H1_LPE_SCORE_UPDATE_INTERVAL` | 分数更新间隔（访问次数） | `8` |
| **准入控制** | `H1_LPE_ADMISSION_MODE` | `diagnostic` / `strict`（仅 diagnostic 生效） | `diagnostic` |
| **输出** | `EDGEKV_H1_STATS_DIR` | GPU 统计 JSON 输出目录 | 实验 out 子目录 |
| | `EDGEKV_H1_RUNTIME_MONITOR` | 是否记录 LPE 运行时 JSONL 日志 | `0` |
| | `EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES` | 是否包含完整对象画像（大） | `0` |

---

### `_runner.py` — 共享实验辅助库

供各 `run_*.py` 脚本 import 使用的公共函数库，不直接运行。提供统一的实验执行、日志、汇总和清理能力。

**核心函数**：

| 函数 | 参数 | 说明 |
|------|------|------|
| `run_real_cell` | `out_dir`, `visible_devices`, `cli_args`, `env_overrides`, `log_file` | 通过 conda 执行 `run_h1_vllm0110_real.py`，返回退出码 |
| `run_bench_cell` | `out_dir`, `visible_devices`, `env_overrides`, `log_file`, `echo`, `force` | （旧版）执行 `.sh` 包装器 |
| `summarize` | `script_name`, `args` | 运行 `h1/` 下的汇总脚本 |
| `validate_d3_for_lpe_cells` | `base_out` | 自动发现所有 LPE cell 并运行 `validate_d3.py` |
| `cleanup_dirs` | `base_out`, `keep`, `only`, `extra` | 清理中间文件，保留汇总结果 |

---

### `run_test.py` — 快速压力测试启动器

用于快速验证配置，运行纯 ShareGPT LRU 预算扫描（5 个预算点：0.75~0.95）。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visible-devices` | GPU 设备 | `0,1` |
| `--num-prompts` | 请求数 | `256` |
| `--tier` | 实验标签 | `sharegpt_lru_budget5_maxlen1024` |
| `--replay-trace` | trace 路径 | `data/edgekv_traces/source_ablation/sharegpt.jsonl` |
| `--force` | 重跑 | `False` |
| `--out-dir` | 输出子目录 | `/run_test` |

**用法**：

```bash
python h1/run_test.py --visible-devices 0 --num-prompts 128 --force
```

---

### `test_h1_rag_trace.py` — 单元测试（无 CLI）

直接运行 `pytest` 或 `python -m unittest` 执行，验证 trace 加载、字段映射、`sitecustomize` 策略钩子等核心功能。

**用法**：

```bash
pytest h1/test_h1_rag_trace.py -v
# 或直接运行
python h1/test_h1_rag_trace.py
```

---

## 启动器调用关系图

下图展示了所有 `run_*.py` 启动器之间的调用层级关系（箭头方向表示调用或调度依赖）：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户驱动层 (User Entry)                           │
│  run_test.py, run_find_hit.py, run_find_interval_2.py,                     │
│  run_step3_repeat.py, run_step3_real.py, run_step3_budget_tiers.py        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       实验编排层 (Orchestration)                           │
│                                                                             │
│  run_find_hit.py ───► scripts/optimize_h0_pressure_trace.py (生成 trace)   │
│       │                                                                     │
│       └──► run_step3_budget_tiers.py (扫描多预算)                          │
│                                                                             │
│  run_find_interval_2.py ───► run_step3_budget_tiers.py (粗网格扫描)         │
│                                                                             │
│  run_step3_budget_tiers.py ───► _runner.run_real_cell()                     │
│                                                                             │
│  run_step3_repeat.py ───► (循环) run_step3_budget_tiers.py (多 rep)         │
│       │                                                                     │
│       └──► summarize_step3_repeat.py                                       │
│       └──► validate_d3.py                                                  │
│                                                                             │
│  run_step3_real.py ───► build_h0_replay_trace.py (预生成 trace)             │
│       │                                                                     │
│       ├──► (循环) run_h1_vllm0110_real.py (多 rep × 策略 × 预算)            │
│       │                                                                     │
│       └──► summarize_step3_real.py                                         │
│       └──► visualize_step3_real.py                                         │
│       └──► validate_d3.py                                                  │
│                                                                             │
│  run_test.py ───► run_step3_budget_tiers.py (纯 LRU 快速扫描)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         执行引擎层 (Execution)                             │
│                                                                             │
│  _runner.run_real_cell()  ───►  conda run python run_h1_vllm0110_real.py   │
│  _runner.run_bench_cell() ───►  bash run_h1_policy_serving_bench.sh        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       底层核心 (Core Engine)                               │
│                                                                             │
│  run_h1_vllm0110_real.py                                                   │
│       │                                                                     │
│       ├──► 加载 trace (复用 h0/run_h0_vllm 的加载逻辑)                     │
│       ├──► 构造 LLM 实例 (vLLM 0.11.0)                                    │
│       ├──► 调用 sitecustomize.py 注入策略                                  │
│       └──► 输出 *_summary.json, *_requests.csv, edgekv_gpu_stats/          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键说明**：

1. **`run_find_hit.py` 与 `run_find_interval_2.py`**：核心都是调用 `run_step3_budget_tiers.py` 完成实验，区别在于前者额外处理 trace 生成和曲线判断，后者专注于定档最近点。
2. **`run_step3_repeat.py` 与 `run_step3_real.py`**：都遵循“多 rep → 汇总 → 可视化”流水线；`run_step3_real.py` 专用于真实混合 trace，支持并发扫描（`--batch-sweep`）；`run_step3_repeat.py` 用于预定义压力工作点。
3. **所有路径的终点**：最终都汇聚到 `run_h1_vllm0110_real.py`（或旧版 `.sh` 包装器），它才是真正与 vLLM 交互并产出原始数据的核心。
4. **汇总/可视化/验证**：在完成实验矩阵后，各启动器会自动调用 `summarize_*.py`、`visualize_*.py` 和 `validate_d3.py` 生成最终报告，做到端到端自动化。

---

## 环境要求

所有实验均在 **`edgekv-vllm0110`** conda 环境下运行，该环境包含：

- Python 3.10+
- vLLM 0.11.0
- PyTorch 2.3+
- transformers
- 其他依赖见 `environment.yml`

激活环境：

```bash
conda activate edgekv-vllm0110
```

如需 dry-run 模式（仅打印命令，不实际执行）：

```bash
EDGEKV_DRY_RUN=1 python h1/run_step3_real.py
```