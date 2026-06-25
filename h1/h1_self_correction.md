# H1 self-correction: LPE 从表现不佳到修正实验设计

## 0. 实验目标与范围

H1 的目标是在 GPU-only vLLM v1 路径上比较 GPU prefix-cache 驱逐策略：

- `vllm_default`
- `h1_lru`
- `h1_lfu`
- `h1_lpe`

所有自定义策略通过 `h1/sitecustomize.py` monkey-patch vLLM v1 `BlockPool` 的 lookup、admission、eviction 和 touch 路径。H1 只研究 GPU prefix cache eviction，不进入 P2 的 RRS/offload、质量层级或 CPU cache。

核心问题是：

1. 在有 GPU KV cache 压力时，LPE 是否稳定优于 LRU。
2. TTFT 是否与显存预算呈可解释相关性。
3. 如果 LPE 表现不佳，原因是策略实现开销、实验设计问题，还是 LPE score 本身没有区分信号。

最终实验纪律：任何策略结论必须至少重复 3 次并取中位数；单次 p95 TTFT 不作为主证据。

## 1. 第一次测试：LPE 表现不佳

最初实验在 `prefix_repetition` workload 上比较 LPE、LRU、LFU 和 vLLM default。该 workload 的 prefix 复用近似均匀，所有 prefix 的复用概率接近，理论上不利于 LPE 依赖 score 区分冷热对象。

低方差、最可靠的压力点为：

```text
gpu_mem_util=0.720
num_prompts=200
num_prefixes=8
prefix_len=512
suffix_len=128
output_len=1
request_rate=18.5
```

该点的多次测量中位数为：

| 策略 | p95 TTFT | hit_rate | evictions | throughput |
| --- | ---: | ---: | ---: | ---: |
| LRU | 1028 ms | 0.9686 | 1256 | 18.26 |
| vLLM default | 1024 ms | - | - | 18.27 |
| LFU | 1222 ms | 0.9667 | 1734 | 16.95 |
| LPE | 1148 ms | 0.9674 | 1595 | 17.67 |

结论：LPE 比 LRU 慢约 11.7%，比 vLLM default 慢约 12%，且区间不重叠。LFU 也更差。LRU 与 vLLM default 基本无法区分。

另外两个现象后来被证明是单次运行方差：

- `gpu_mem_util=0.730, n=200` 下 vLLM default 首测 798 ms，复测回到约 1023 ms。
- `gpu_mem_util=0.710, n=400` 饱和点中，Step2 单次显示 LPE 比 LRU 快 11.1%，但 3 次中位数变为 LPE 仅比 LRU 慢 1.5%，区间重叠。

因此第一次结论不是“LPE 完全无效”，而是：在当前均匀复用 workload 上，LPE 不能稳定优于 LRU，并且单次 p95 TTFT 方差足以制造错误判断。

## 2. 归因：LPE 为什么输给 LRU

主要原因不是策略 CPU 开销，而是 LPE 的 score reorder 在无冷热区分的 workload 上制造了 churn。

在 `prefix_repetition` 中，各 prefix 复用概率近似均匀。LPE 使用：

```text
score = p_reuse * c_recomp / size
```

对 free queue 中的候选块排序，优先驱逐低 score 块。但当 score 主要由 size 或噪声决定，而不是由真实冷热差异决定时，这种排序等价于扰乱 LRU 原本干净的 recency 顺序。

关键诊断数据：

| 指标 | LRU | LPE | 解释 |
| --- | ---: | ---: | --- |
| evictions | 1256 | 1595 | LPE 多驱逐约 27% |
| cached_blocks | 1742 | 2082 | LPE 多缓存约 20% |
| hit_rate | 0.9686 | 0.9674 | 命中率没有改善 |
| hot_prefix_evictions | - | 261 | LPE 误驱逐热 prefix |

也就是说，LPE 多做了 evict -> re-admit 循环，却没有提高 hit rate，最终增加 re-prefill 和尾延迟。

决定性对照是关闭 LPE reorder：

| 指标 | LRU | LPE reorder on | LPE reorder off |
| --- | ---: | ---: | ---: |
| p95 TTFT | 1028 | 1148 | 1032 |
| evictions | 1256 | 1595 | 1316 |
| hit_rate | 0.9686 | 0.9674 | 0.9686 |
| hot_prefix_evictions | - | 261 | 2 |

关闭 reorder 后，LPE 立即退化为近似 LRU，TTFT 和 hit_rate 都与 LRU 持平。这说明劣化来自 score reorder，而不是不可控的 vLLM 接线问题。

策略开销是次要问题：在 0.730/n200 下，LPE 的 `policy_time_ms_total` 约 176 ms，平均 20.6 us；LRU 约 44 ms，平均 5.6 us。这个差距存在，但在已测点没有单独解释 p95 TTFT 的 100 ms 级劣化。

## 3. 三步走修正

### Step 1：测试哪些参数影响 TTFT

先只使用 `h1_lru` 做单因素实验，避免策略差异干扰 workload 校准。基础配置为：

```text
policy=h1_lru
dataset=prefix_repetition
num_prompts=160
request_rate=18.5
num_prefixes=8
prefix_len=512
suffix_len=128
output_len=1
gpu_memory_utilization=0.710
```

五个参数的影响排序大致为：

```text
suffix_len ~= num_prefixes > prefix_len > num_prompts > request_rate
```

主要观察：

- `request_rate` 主要影响排队和调度压力，hit_rate 与 evictions 基本不变。
- `num_prompts` 增大时 unique suffix KV footprint 增加，evictions 和 p95 TTFT 同步上升。
- `num_prefixes` 同时增加 shared prefix footprint 并降低单 prefix 复用集中度，`num_prefixes=16` 会明显过载。
- `prefix_len` 同时增加 prefill 成本和 prefix KV footprint，512 之后 TTFT 增长明显。
- `suffix_len` 直接增加未缓存 prefill 和 unique KV footprint，`suffix_len>=160` 会进入明显饱和。

校准结果显示 `request_rate=18.5` 接近目标压力区间：

| rr | p95 TTFT | hit_rate | evictions | throughput |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 917.09 | 0.968392 | 887 | 16.55 |
| 18.5 | 1056.34 | 0.968392 | 887 | 16.62 |
| 19 | 1204.52 | 0.968392 | 887 | 16.66 |

后续策略比较推荐使用：

```text
num_prompts=160 或 192
request_rate=18.5
num_prefixes=8
prefix_len=512
suffix_len=128
output_len=1
```

避免使用 `num_prefixes=16` 或 `suffix_len>=160` 作为主对比点，因为这些点已混入明显系统饱和和排队影响。

### Step 2：测试最优参数组合

Step2 横向比较多组 workload，目标是找出最能体现 LPE 相对 LRU 提升的参数环境，而不是直接定最终预算。

实验矩阵：

```text
5 workloads * 2 policies(h1_lru, h1_lpe) * 3 budgets = 30 runs
budgets = tight(0.710), mid(0.735), loose(0.774)
num_prompts=400
output_len=1
request_rate=18.5
```

tight 档下 LPE 相对 LRU 的 p95 TTFT gain：

| workload | tight p95 gain | hit_rate delta | eviction delta | tight 是否饱和 |
| --- | ---: | ---: | ---: | --- |
| n224_p8_pl512_s128 | +12.21% | +0.0076 | -1097 | 是 |
| n200_p8_pl512_s128 | +11.10% | +0.0066 | -998 | 是 |
| n200_p8_pl640_s128 | +7.98% | +0.0041 | -354 | 是 |
| base_n200_p8_pl512_s128 | +6.67% | +0.0066 | -1018 | 是 |
| n200_p12_pl512_s128 | +4.63% | +0.0121 | -999 | 是 |

Step2 的问题是：LPE 优势集中在 tight 档，但 tight 档全部饱和，throughput 只有约 8-12 req/s；mid/loose 又压力不足，LPE 与 LRU 基本持平甚至略差。

因此 Step2 给出的修正方向是：

- 固定 prefix 结构为 `num_prefixes=8 / prefix_len=512 / suffix_len=128`。
- 将 `num_prompts` 从 400 降到 200，避免直接饱和。
- 不再直接使用 0.710/0.735/0.774 作为最终三档，而是扫描 `gpu_memory_utilization`，寻找“有 eviction 但未饱和”的甜区。

### Step 3：进行最终测试

最终主路径改为 serving-bench，而不是 real-replay。

serving-bench 使用：

```text
vllm serve + vllm bench serve + request_rate
```

它包含真实请求到达率、排队和 continuous batching，报告真实 p95 TTFT。最终测试采用三个工作点：

| 场景 | 配置 | 定位 |
| --- | --- | --- |
| A | `gpu_mem_util=0.720, n=200` | 有压力非饱和，低方差，主证据 |
| B | `gpu_mem_util=0.730, n=200` | 非饱和，策略基本不可区分 |
| C | `gpu_mem_util=0.710, n=400` | 饱和，高方差，不作主证据 |

固定 workload：

```text
prefix_repetition
num_prefixes=8
prefix_len=512
suffix_len=128
output_len=1
request_rate=18.5
```

主入口：

```bash
python3 h1/run_step3_repeat.py --visible-devices 0,1
```

该脚本执行 A/B/C 三个场景、4 个策略、N 次重复，并对每个 cell 跨 rep 取中位数。已有 `aggregate.csv` 的 cell 会自动跳过，支持断点续跑。

最终结论：

- 场景 A 是主证据：LPE 稳定慢于 LRU，原因是均匀复用下 score reorder 制造 churn。
- 场景 B 策略无法有效区分，证明非饱和压力不足。
- 场景 C 方差过高，不适合作主结论。

## 4. real-replay 路径的修正定位

real-replay 路径最初用于 ShareGPT + HotpotQA 混合 trace，但 `step3_real` 中三档预算、四种策略的 p95 全部挤在 1100-1200 ms，几乎不随预算或策略变化。

诊断结论：预算确实正确传入 vLLM，问题在 harness 结构。

1. 记录的 `ttft_proxy_ms` 实际是离线 `LLM.generate()` 一整批的墙钟耗时，不是真实 time-to-first-token。
2. 默认 `replay_batch_size=2`，同时 `max_num_seqs=2`，活跃 KV 需求很小，tight 预算也能装下。
3. replay 是串行分批，没有请求到达率、排队或 continuous batching，因此不能复现 serving-bench 中 TTFT 爆炸的机制。

real-replay 的修改方向：

- 方案一：把 `replay_batch_size` 提高到 64-256，并提供 `--batch-sweep 2 32 64 128`，使预算开始咬合。但该路径仍只能称为批延迟代理。
- 方案二：主结论切到 serving-bench。该路径报告真实 p95 TTFT，是 H1 最终主证据。

因此当前文档将 real-replay 定位为次要/对照路径，主结论只使用 serving-bench。

## 5. LPE 实现优化

LPE 的实现优化围绕两个方向：降低策略开销，以及避免无区分度时乱重排。

已明确或已落地的优化方向包括：

- `h1_lru` 改为原生 LRU 顺序加诊断计数，去掉冗余 recency reorder。
- LPE 支持 `H1_LPE_REORDER_MODE=off`，在均匀复用 workload 上可退化为近似 LRU。
- COP 改为对象级画像，KV cache size 使用 resident block 的实际页大小累计。
- 保留策略计时、score 分布、hot prefix eviction 等诊断字段，用于定位 LPE 劣化来源。

P0/P1 优化计划：

| 阶段 | 目标 | 核心动作 |
| --- | --- | --- |
| P0-1 | 降低 free queue 全量排序开销 | 增加 reorder 计时与计数；按压力触发；使用候选窗口排序；支持 `window/full/off` |
| P0-2 | 压力感知 fast path | 低压力时不重排、不刷新全量 score，只维护 O(1) 画像 |
| P1-1 | 完善 `p_reuse` | 引入 freq、recency、object-type prior，避免单一命中率退化 |
| P1-2 | 完整生命周期 LPE | admission、touch、eviction、pin/drop、TTL aging 全链路诊断 |
| P1-3 | RAG/session 闭环 | 统一 prefix、RAG chunk、session 的对象 ID、block 映射和命中反馈 |
| P1-4 | 诊断与回归门槛 | 输出 score、evicted score、hot eviction、policy time、reorder time |

H1 内不实现 P2：

- RRS/offload
- `c_restore`、带宽、反序列化成本
- 质量层级或 TMS

## 6. 当前统一实验入口

主路径：serving-bench，出最终结论。

```bash
python3 h1/run_step3_repeat.py --visible-devices 0,1
```

可选参数：

```text
--reps N
--force
--keep-cells
EDGEKV_DRY_RUN=1
```

可视化：

```bash
python3 h1/visualize_lpe_scenarios.py
```

输出：

```text
h1/out/step3_repeat/step3_repeat_summary.csv
h1/out/step3_repeat/lpe_scenarios.png
h1/out/step3_repeat/lpe_scenarios.pdf
```

次要路径：real-replay + 并发扫描。

```bash
python3 h1/run_step3_real.py --visible-devices 0,1
python3 h1/run_step3_real.py --batch-sweep 2 32 64 128
```

real-replay 输出的 `ttft_proxy_ms` 是批延迟代理，不作为真实 TTFT 主结论。

## 7. 最终判断

H1 的自我修正过程可以概括为：

1. 第一次测试发现 LPE 没有稳定优于 LRU，甚至在可靠压力点上更慢。
2. 归因实验确认劣化来自 score reorder 在均匀复用负载上制造 churn，而不是接线错误或单纯 CPU 开销。
3. 单因素实验确定 TTFT 的关键 workload 参数，并排除明显饱和点。
4. 参数组合实验发现 tight 档虽显示 LPE 收益，但全部饱和，不能作为主证据。
5. 最终实验切到 serving-bench 的三场景重复测量，用低方差的有压力非饱和点作为主结论。
6. real-replay 被降级为批延迟代理路径，只用于对照或并发扫描。
7. LPE 实现优化重点转为压力感知、候选窗口排序、冷热区分度门槛和完整诊断。

当前最重要的结论是：在 `prefix_repetition` 这种近似均匀复用 workload 上，LPE 的 score 没有足够冷热区分信号，reorder 会扰乱 LRU 的 recency 顺序并造成 churn。因此该负结果不能否定 LPE 的设计，只说明公平验证 LPE 需要偏斜复用 workload，或者在 score 区分度不足时自动退化为 LRU。
