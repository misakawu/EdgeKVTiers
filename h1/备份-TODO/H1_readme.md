# H1 Readme

## 项目定位

H1 是 EdgeKVTiers 里针对 GPU prefix cache 驱逐策略的实验目录，核心目标是比较 `vllm_default`、`h1_lru`、`h1_lfu`、`h1_lpe` 在不同 workload 和显存预算下的 TTFT、hit rate 和 eviction 表现。

当前 H1 的主结论路径是 `serving-bench`，`real-replay` 只作为对照路径。

## 目录结构

```text
h1/
├── H1_readme.md
├── h1_self_correction.md
├── H1_复盘与重跑配置单.md
├── _runner.py
├── sitecustomize.py
├── run_step3_budget_tiers.py
├── run_step3_repeat.py
├── run_step3_real.py
├── summarize_step3_budget_tiers.py
├── summarize_step3_repeat.py
├── summarize_step3_real.py
├── aggregate_h1_serving_bench.py
├── visualize_lpe_scenarios.py
├── visualize_step3_real.py
├── run_h1_vllm0110_real.py
├── test_h1_rag_trace.py
└── out/
```

## 文件分工

### 1. 实验主入口

- `run_step3_repeat.py`
  - 最终主入口。
  - 跑 A/B/C 三个场景，比较 4 个策略，并对重复实验取中位数。
  - 对应最终测试和主结论输出。

- `run_step3_budget_tiers.py`
  - 单个工作点的 budget × policy 矩阵执行器。
  - 被 `run_step3_repeat.py` 复用。
  - 对应 Step 3 的预算标定和单点策略对比。

- `run_step3_real.py`
  - real ShareGPT+HotpotQA replay 路径。
  - 支持预算扫描和并发扫描。
  - 结果是 `ttft_proxy_ms`，属于批延迟代理，不是主结论。

### 2. 底层执行器

- `run_h1_vllm0110_real.py`
  - real-replay harness。
  - 负责把 trace 喂给 vLLM，记录每个 cell 的 summary。

- `sitecustomize.py`
  - H1 的核心实现文件。
  - 在 vLLM 进程启动时自动加载，patch GPU prefix cache 的策略逻辑。
  - 包含 LRU / LFU / LPE 的驱逐、重排、score、诊断计数。

- `_runner.py`
  - 所有 `run_step3_*.py` 的共享工具。
  - 封装 cell 运行、summary 调用、日志、清理和 dry-run。

### 3. 汇总与可视化

- `aggregate_h1_serving_bench.py`
  - 把 serving-bench 输出的 `result.json` 和 GPU stats 汇总成 `aggregate.csv`。

- `summarize_step3_budget_tiers.py`
  - 汇总单个 budget-tier 目录。
  - 生成 budget/policy 粒度的 summary CSV。

- `summarize_step3_repeat.py`
  - 对 A/B/C 三个场景跨 rep 取中位数。

- `summarize_step3_real.py`
  - 对 real-replay 的 budget/policy/batch-size 结果做中位数汇总。

- `visualize_lpe_scenarios.py`
  - 画 A/B/C 三个场景的主图。

- `visualize_step3_real.py`
  - 画 real-replay 的 4 策略 × 3 预算图。

### 4. 测试与验证

- `test_h1_rag_trace.py`
  - H1 的 smoke test / 单测。
  - 主要检查 trace 解析、RAG/session 字段、LPE score 和对象识别是否正确。

## 当前实验主线

1. 先用 `run_step3_budget_tiers.py` 和单参数实验思想确定 workload 的压力点。
2. 再用 `run_step3_repeat.py` 跑最终主路径，得到稳定的中位数结论。
3. `run_step3_real.py` 只做 real-replay 对照，不作为真实 TTFT 主证据。
4. `sitecustomize.py` 是策略实现和优化的唯一关键位置。

## 启动方式

从仓库根目录启动：

```bash
cd /DATACENTER3/zhenxiang.wang/work/EdgeKVTiers
```

高层脚本会通过 `conda run --no-capture-output -n edgekv-vllm0110` 启动底层 vLLM cell，并自动设置 `PYTHONPATH=.:h1:h0`、`CUDA_VISIBLE_DEVICES`、`VLLM_USE_V1=1` 和 `EDGEKV_H1_GPU_POLICY`。通常不需要手动激活 conda 环境。

先用 dry-run 检查命令：

```bash
EDGEKV_DRY_RUN=1 python3 h1/run_step3_real.py \
  --recommended-batch-sweep \
  --reps 1 \
  --budgets tight \
  --policies h1_lru \
  --max-requests 8 \
  --visible-devices 0,1 \
  --keep-cells
```

常用启动命令：

```bash
# 快速 smoke test：LRU/LPE, tight/mid, 保留 cell 输出。
python3 h1/run_test.py --visible-devices 0,1 --num-prompts 64 --force

# 主重复实验：三档压力点 × 四策略 × reps，输出跨 rep 中位数。
python3 h1/run_step3_repeat.py --visible-devices 0,1

# 单个 budget-tier 矩阵：适合调参和局部复跑。
python3 h1/run_step3_budget_tiers.py \
  --visible-devices 0,1 \
  --budgets "tight mid loose" \
  --policies "h1_lru h1_lfu vllm_default h1_lpe"

# real-replay 并发扫描：先用 8/16/32/64 选工作点。
python3 h1/run_step3_real.py --batch-sweep 8 16 32 64 --visible-devices 0,1

# real-replay 单点：默认推荐并发 32。
python3 h1/run_step3_real.py --visible-devices 0,1 --replay-batch-size 32

# length_bucket 对照：比较同一并发下 batch 内排队跨度。
python3 h1/run_step3_real.py \
  --visible-devices 0,1 \
  --replay-batch-size 32 \
  --batch-order length_bucket \
  --warmup-batches 1
```

底层 harness 也可以直接运行：

```bash
PYTHONPATH=.:h1:h0 conda run --no-capture-output -n edgekv-vllm0110 \
  python h1/run_h1_vllm0110_real.py \
  --policies h1_lru \
  --budgets tight \
  --max-requests 128 \
  --replay-batch-size 32 \
  --visible-devices 0,1
```

常用输出：

```text
h1/out/run_test/
h1/out/step3/<tier>/step3_summary.csv
h1/out/step3_repeat/step3_repeat_summary.csv
h1/out/step3_real/step3_real_summary.csv
```

real-replay 的排队诊断重点看：

```text
queue_wait_ratio_mean
queue_wait_p95_ms
prefill_p95_ms
batch_queue_span_p95_ms
```

## 关键文档

- [h1_self_correction.md](./h1_self_correction.md): 记录从第一次失败到最终修正的完整过程。
- [H1_复盘与重跑配置单.md](./H1_复盘与重跑配置单.md): 面向复盘和重跑的高层配置说明。

## 输出目录

所有实验结果默认写到 `h1/out/` 下，常见子目录包括：

- `h1/out/step3/`
- `h1/out/step3_repeat/`
- `h1/out/step3_real/`

这里会存放 `aggregate.csv`、summary CSV、图像和日志。
