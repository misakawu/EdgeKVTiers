# EdgeKVTiers

EdgeKVTiers 是一个面向大模型 KV cache 分层、offload 与驱逐策略评估的实验仓库。当前重点是 H1：在真实 vLLM 0.11.0 serving 环境中比较不同 GPU prefix-cache / KV offload 策略对 TTFT 的影响。

仓库中同时保留了 H0 trace 构造与回放、H1 vLLM 真实实验、H1 serving benchmark、H2O/KIVI baseline 环境以及预实验脚本。

## 仓库结构

```text
.
├── h0/                         # H0 trace 构造、vLLM 回放、smoke test
├── h1/                         # H1 vLLM 0.11.0 策略接入、benchmark 与实验驱动
│   ├── run_step3_repeat.py     # 【主入口】serving-bench A/B/C × 4 策略 × N reps
│   ├── run_step3_budget_tiers.py
│   ├── run_step3_real.py       # 【次要入口】real-replay + 并发扫描
│   ├── run_h1_policy_serving_bench.sh   # serving-bench cell 执行器
│   ├── run_h1_vllm0110_real.py          # real-replay harness
│   ├── aggregate_h1_serving_bench.py
│   └── sitecustomize.py        # GPU prefix-cache 策略实现（LRU/LFU/LPE）
├── pre实验/                    # 仿真/预实验脚本
├── third_party/                # H2O、KIVI 等第三方源码
├── models/                     # 本地模型目录
└── 论文规划/                   # 论文规划与实验设计文档
```

> H1 实验驱动（原 `h1-pre/`）已并入 `h1/`，详见 `h1/README.md`。

## 当前 H1 实验目标

H1 的主要目标是评估不同 KV/cache 策略对在线 TTFT 的影响。当前关注的策略包括：

- `h1_lru`：LRU 驱逐策略
- `h1_lfu`：LFU 驱逐策略，LRU 作为 tie break
- `h1_lpe`：基于 `score = p_reuse * c_recomp / size` 的 LPE 策略
- `vllm_default`：vLLM 默认行为，主要用于真实矩阵实验

当前 serving benchmark 优先使用 `prefix_repetition` workload，因为它能显式控制 unique KV footprint：

```text
unique_kv_tokens ~= num_prefixes * prefix_len + num_prompts * suffix_len
```

推荐先把 LRU p95 TTFT 校准到约 `1100 ms`，再复用同一 workload 比较 LFU、vLLM default 和 LPE。

## 环境要求

当前已验证路径和环境以本机为例：

```text
项目目录: /DATACENTER3/zhenxiang.wang/work/EdgeKVTiers
Conda:    /DATACENTER3/zhenxiang.wang/miniforge3
数据目录: /DATACENTER3/zhenxiang.wang/data
GPU:      NVIDIA GeForce RTX 2080 Ti
```

H1 真实 vLLM 0.11.0 实验使用 Conda 环境：

```text
edgekv-vllm0110
```

该环境中已验证的关键组件：

```text
vllm==0.11.0
torch==2.8.0+cu128
torchvision==0.23.0
torchaudio==2.8.0
```

基础系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential curl
git lfs install
```

## 模型和数据

默认模型：

```text
models/Qwen2.5-7B-Instruct
models/facebook_opt_125m
```

Qwen 模型可通过 Hugging Face 下载：

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

如 Hugging Face 网络不可用，可以使用 ModelScope：

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir models/Qwen2.5-7B-Instruct
```

ShareGPT 默认数据路径：

```text
/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

HotpotQA 默认数据路径：

```text
/DATACENTER3/zhenxiang.wang/data/hotpotqa
```

## H1 Serving Benchmark

### 单策略测试

`h1/run_h1_policy_serving_bench.sh` 用于测试单个策略。常用环境变量如下：

```text
H1_GPU_POLICY                         策略名，默认 h1_lru
H1_BENCH_NUM_PROMPTS                  请求数量，默认 128
H1_BENCH_REQUEST_RATE                 请求到达速率，默认 inf
H1_BENCH_MAX_CONCURRENCY              最大并发，可留空
H1_GPU_MEMORY_UTILIZATION             vLLM GPU memory utilization，默认 0.710
H1_PREFIX_REPETITION_NUM_PREFIXES     prefix 组数，默认 8
H1_PREFIX_REPETITION_PREFIX_LEN       prefix 长度，默认 512
H1_PREFIX_REPETITION_SUFFIX_LEN       suffix 长度，默认 128
H1_PREFIX_REPETITION_OUTPUT_LEN       输出长度，默认 1
```

示例：使用 LRU 校准 p95 TTFT。

```bash
H1_GPU_POLICY=h1_lru \
H1_BENCH_NUM_PROMPTS=160 \
H1_BENCH_REQUEST_RATE=18.5 \
H1_PREFIX_REPETITION_NUM_PREFIXES=8 \
H1_PREFIX_REPETITION_PREFIX_LEN=512 \
H1_PREFIX_REPETITION_SUFFIX_LEN=128 \
H1_PREFIX_REPETITION_OUTPUT_LEN=1 \
H1_GPU_MEMORY_UTILIZATION=0.710 \
bash h1/run_h1_policy_serving_bench.sh h1/out/h1_lru_calib_rr18_5_n160_s128 0,1
```

输出聚合结果：

```text
h1/out/h1_lru_calib_rr18_5_n160_s128/aggregate.csv
```

### LPE 三档显存预算 / 多策略对比

三档 GPU memory budget：

```text
tight: gpu_memory_utilization=0.710
mid:   gpu_memory_utilization=0.735
loose: gpu_memory_utilization=0.774
```

策略对比不再单独用一个 shell 脚本,而是经 `h1/` 的实验驱动统一编排（≥3 次取中位数）：

```bash
# 主路径（serving-bench A/B/C 场景,真实 p95 TTFT,出主结论）：
python3 h1/run_step3_repeat.py --visible-devices 0,1

# 次要路径（real-replay 重放,ttft 为批延迟代理,作对照）：
python3 h1/run_step3_real.py --visible-devices 0,1
```

详见 `h1/README.md`。底层单 cell 仍由 `h1/run_h1_policy_serving_bench.sh` 执行。

### 结果字段

每次 serving benchmark 的核心结果在：

```text
h1/out/<run_name>/aggregate.csv
```

重点字段：

```text
p95_ttft_ms
p50_ttft_ms
mean_ttft_ms
request_throughput
hit_rate
gpu_prefix_cache_lookup_total
gpu_prefix_cache_lookup_hits
gpu_prefix_cache_lookup_misses
gpu_prefix_cache_evictions
gpu_prefix_cache_cached_blocks
gpu_prefix_cache_queue_reorders
avg_p_reuse
avg_score
```

判断思路：

- `p95_ttft_ms` 上升但 `evictions/hit_rate` 基本不变，通常说明排队或调度压力变大。
- `p95_ttft_ms` 与 `evictions/hit_rate/cached_blocks` 同时变化，通常说明 KV cache 容量或驱逐策略产生影响。
- `request_throughput` 明显低于设定 request rate 时，说明系统已经饱和，TTFT 混入了明显排队影响。

## H1 预实验方案

详细的 TTFT 参数实验方案在：

```text
h1/ttft_parameter_experiment_plan.md
```

该方案要求 5 个参数分步测试，每次只改变一个因素：

```text
request_rate
num_prompts
num_prefixes
prefix_len
suffix_len
```

`output_len` 固定为 1，因为 TTFT 只关注首 token；增大输出长度主要影响 decode 和 e2e latency。

推荐执行顺序：

1. 先用 `h1_lru` 校准到 p95 TTFT 约 `1100 ms`。
2. 分别测试 5 个参数对 p95 TTFT 的影响。
3. 选择稳定且有适度 eviction 的 workload。
4. 测试 tight/mid/loose 三档显存预算。
5. 在同一 workload 下比较 LRU、LFU、vLLM default 和 LPE。

## H1 真实 vLLM 矩阵实验（real-replay,次要路径）

`h1/run_step3_real.py` 把 ShareGPT+HotpotQA 冻结成一条 replay trace,再跑：

```text
4 policies x 3 budgets x N reps
```

策略 `vllm_default / h1_lru / h1_lfu / h1_lpe`，预算 `tight / mid / loose`。

```bash
# 三档 × 四策略矩阵（默认并发 64）：
python3 h1/run_step3_real.py --visible-devices 0,1

# h1-report.md 方案一并发扫描:固定单一 budget,定位预算开始咬合的并发点：
python3 h1/run_step3_real.py --batch-sweep 2 32 64 128
```

trace 由 `h0/build_h0_replay_trace.py` 生成并只解析一次。输出与汇总在 `h1/out/step3_real/`
（`step3_real_summary.csv`）。注意 `ttft_proxy_ms` 是整批 `LLM.generate()` 的墙钟（**批延迟代理,
非真实 TTFT**）,主结论以 serving-bench 路径为准,原委见 `h1/h1-report.md`。

## H1 策略实现说明

H1 vLLM 0.11.0 策略适配代码在：

```text
h1/sitecustomize.py
```

其中：

- vLLM 进程通过 `PYTHONPATH=h1:h0` 自动加载 `sitecustomize.py`，monkey-patch v1 GPU prefix cache。
- `h1_lru` 实现 LRU；`h1_lfu` 实现 LFU，并用 LRU 处理频次相同的块。
- `h1_lpe` 使用预估复用概率和重算成本计算驱逐分数（`score = p_reuse * c_recomp / size`）。
- patch 导出 lookup、hit、eviction、reorder 等统计供 `aggregate_h1_serving_bench.py` 聚合。

H1 serving benchmark 通过如下环境变量控制策略：

```text
EDGEKV_H1_GPU_POLICY=h1_lru|h1_lfu|h1_lpe
```

脚本会自动设置该变量；通常不需要手动设置。

## H0 Trace 与回放

H0 用于构造 ShareGPT + HotpotQA mixed replay trace，并可对 vLLM server 发起回放。

构造 trace 示例：

```bash
PYTHONPATH=h0 python h0/build_h0_replay_trace.py \
  --trace-path /DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --hotpotqa-path /DATACENTER3/zhenxiang.wang/data/hotpotqa \
  --out /DATACENTER3/zhenxiang.wang/data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions_pressure.jsonl \
  --workload mixed \
  --max-sessions 200 \
  --max-requests 1024 \
  --rag-requests 100 \
  --hotpotqa-max-examples 5 \
  --sharegpt-order longest
```

H1 真实矩阵实验默认复用该 trace。

## 第三方 Baseline

第三方源码放在：

```text
third_party/H2O
third_party/KIVI
```

固定版本：

```text
H2O:  ac75c2a8a9e76832b2a4139b9363373b56336bfb
KIVI: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

KIVI 环境通常使用 `edgekv-kivi`，H2O 环境通常使用 `edgekv-h2o`。这两个 baseline 与 H1 vLLM 0.11.0 实验相互独立，避免依赖冲突。

## 常见问题

### 1. 为什么 TTFT 会随 request_rate 大幅变化？

`request_rate` 会直接改变请求排队和调度压力。即使 KV cache hit rate 不变，p95 TTFT 也可能因为排队变长而上升。因此校准阶段必须先固定 workload，只扫 request rate。

### 2. 为什么 output_len 固定为 1？

TTFT 只统计首 token 延迟。增大 `output_len` 会明显增加 decode 和 e2e latency，但不利于隔离 KV cache 对 TTFT 的影响。

### 3. 如何判断显存预算是否真的影响 TTFT？

看 `gpu_prefix_cache_evictions`、`hit_rate`、`cached_blocks` 和 `p95_ttft_ms` 是否同时变化。只看 TTFT 不够，因为高并发排队也会拉高 TTFT。

### 4. 为什么需要 tight/mid/loose 三档？

三档预算对应不同 GPU KV cache 容量。通过构造 unique KV footprint 分别跨过 tight、mid、loose 的容量边界，可以判断策略收益是否来自更少 eviction 或更高 prefix-cache 命中。

### 5. GPU 忙或端口冲突怎么办？

先检查 GPU 进程：

```bash
nvidia-smi
```

serving benchmark 默认端口从 `8100` 开始。可用环境变量改端口：

```bash
H1_SERVE_PORT=8110
```

## 参考实验文档

- `h1/ttft_parameter_experiment_plan.md`：TTFT 参数分步实验方案
- `论文规划/08_实验平台与计划实验组.md`：论文实验规划
- `h0/LOG.md`：H0 相关记录
- `h0/TODO.md`：H0 待办与历史说明
