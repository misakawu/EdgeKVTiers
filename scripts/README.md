# `optimize_h0_pressure_trace.py` 参数对 H1 hit rate 的影响

本文说明 `scripts/optimize_h0_pressure_trace.py` 如何生成 H1 使用的压力 trace，以及各参数如何从代码机制和缓存理论两方面影响 prefix-cache hit rate。H1 的目标是观察 LPE 在多个显存预算下相对 LRU/LFU 是否能保留更有价值的 KV，从而降低 p95 TTFT。因此，理想 trace 不应让 hit rate 在某个 budget 上突然全中或全不中，而应在中低容量区间形成可比较的平缓斜坡。

## 1. 代码生成机制

脚本先构造两类对象：

- ShareGPT 对象：从 ShareGPT 会话里截取第一轮 user 文本，构造成长上下文 prompt。
- HotpotQA/RAG 对象：从 HotpotQA chunk group 构造检索上下文 prompt。

每个对象都会带有稳定的 `reuse_key` 和 prompt 开头的 `object_marker`：

```text
[TRACE_OBJECT sharegpt_hot_0000]
[TRACE_OBJECT sharegpt_cold_0012]
[TRACE_OBJECT hotpotqa_hot_0004]
```

hot 对象会生成多次访问，cold 对象通常只生成一次访问。然后 `budget_ladder_order()` 从所有对象中选出一批 hot key 和 unique cold key，生成核心访问前缀：

```text
prime: 依次访问 N 个 hot 对象的第 1 次访问
repeat scan_probe_rounds 轮:
  扫描 scan_cold_objects 个 unique cold 对象
  复测全部 N 个 hot 对象
append: 追加剩余未使用请求
```

代码约束：

- `--hot-repeats >= --scan-probe-rounds + 1`
- `--rag-hot-repeats >= --scan-probe-rounds + 1`
- hot key 数量必须不少于 `--scan-hot-objects`
- unique cold key 数量必须不少于 `--scan-cold-objects * --scan-probe-rounds`

脚本最后写出 JSONL trace 和 summary。summary 中最关键的结构指标是：

- `estimated_hot_working_set_tokens`：ladder 中被 prime 的 hot 对象首访 prompt token 总量。
- `estimated_cold_scan_tokens_per_round`：每轮 cold scan 的 prompt token 总量。
- `avg_hot_prompt_est_tokens` / `avg_cold_prompt_est_tokens`：hot/cold prompt 的平均估算 token 数。

## 2. hit rate 理论模型

prefix-cache hit rate 由“之前写入的 KV 是否还在缓存中”决定。在 `budget_ladder` 序列中，容量压力来自两个阶段：

- Prime 阶段写入全部 hot 对象 KV，形成 hot working set。
- Cold scan 阶段写入大量只访问一次的 cold KV，挤占缓存空间。
- Probe 阶段再次访问 hot 对象，命中与否取决于对应 hot KV 是否在 cold scan 后仍保留。

因此，hit 曲线随显存预算变化的形状主要由两个量决定：

- hot working set 越大，完整保留全部 hot 对象所需的容量越高，饱和点右移。
- 每轮 cold scan token 越多，越容易把 hot KV 挤出，低容量 hit rate 越低，斜坡起点右移。

如果 hot 对象数量太少，容量跨过某个阈值后会同时保住一大批 hot KV，曲线表现为陡峭台阶。原始预期是：hot 对象数量更多、单对象 token 分布更细时，容量增加会逐步多保留一些 hot KV，曲线更接近平缓斜坡。但本次 A-E 实测表明，在当前 vLLM prefix-cache/block 分配粒度和 `budget_ladder` 访问结构下，单纯增大 hot 对象数或 hot prompt 大小更多是在移动阈值，而不是自然生成平滑过渡。

对 H1 而言，平缓斜坡更有价值：LRU/LFU/LPE 会在多个容量点上做出不同驱逐选择，LPE 的 score 优势才有空间转化成更高 hit rate 和更低 p95 TTFT。

## 3. 本次 A-E 实测结论

固定参数：

- `scan-cold-objects=24`
- `cold-context-words=800`
- `scan-probe-rounds=3`
- `hot-repeats=4`
- `rag-hot-repeats=4`
- `random-seed=2026`

LRU 粗扫结果：

| 组 | scan-hot | hot-context | sharegpt-groups | hot-ratio | rag-requests | hot working set | cold scan/round | hit@0.77 | hit@0.80 | hit@0.83 | hit@0.86 | hit@0.88 | hit@0.90 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| A | 70 | 500 | 200 | 0.30 | 160 | 39,700 | 18.3k/17.1k/17.1k | 0.379 | 0.380 | 0.379 | 0.958 | 0.958 | 0.958 | 陡崖在 0.86 |
| B | 90 | 500 | 260 | 0.32 | 200 | 52,616 | 20.0k/18.7k/19.8k | 0.469 | 0.469 | 0.470 | 0.469 | 0.952 | 0.961 | 陡崖右移到 0.88 |
| C | 110 | 500 | 320 | 0.34 | 240 | 67,245 | 22.0k/20.3k/20.8k | 0.534 | 0.533 | 0.533 | 0.534 | 0.534 | 0.534 | 陡崖被推出 0.90 外，变成中平台 |
| D | 90 | 700 | 260 | 0.32 | 200 | 65,147 | 20.0k/18.7k/19.8k | 0.469 | 0.469 | 0.469 | 0.470 | 0.470 | 0.470 | 增大 hot prompt 后仍是平台 |
| E | 110 | 700 | 320 | 0.34 | 240 | 85,263 | 22.0k/20.3k/20.8k | 0.533 | 0.533 | 0.533 | 0.534 | 0.534 | 0.534 | 更大 hot working set 仍未变平 |

实测修正：

- `scan-hot-objects` 从 70 增到 90 时，低端 hit 从约 0.38 抬到 0.47，陡崖从 0.86 右移到 0.88；它确实移动饱和点，但没有形成多级斜坡。
- `scan-hot-objects` 增到 110 时，0.77-0.90 全部停在约 0.533；这说明高平台被推到 0.90 之外，当前预算网格内只看到一个中间平台。
- `hot-context-words` 从 500 增到 700 时，hot working set 明显增加，但 B->D、C->E 都没有带来平滑斜坡；它主要扩大容量需求，把饱和点继续右移或移出观测区间。
- `sharegpt-groups`、`hot-ratio`、`rag-requests` 在本轮主要提供对象供给。它们伴随 `scan-hot-objects` 增大而增大，但没有独立证据显示能平滑 hit 曲线。
- 当前 `budget_ladder` 的 probe 命中表现更像“容量跨过某个可保留集合阈值后整体跳变”，不是对象数线性增加带来的连续命中。

因此，本轮结论不是“hot 对象还不够多”，而是：在当前访问结构和预算网格下，继续单调扩大 hot working set 只会让陡崖平移；要得到平缓曲线，需要改变容量压力的层次结构或测量网格，而不只是继续放大 hot 集合。

## 4. 调度参数

### `--reuse-schedule`

代码角度：当前唯一可选值是 `budget_ladder`。`validate_args()` 会拒绝其它值，后续固定调用 `budget_ladder_order()` 重排访问序列。

理论影响：

- 它决定 trace 使用 `prime hot -> cold scan -> hot probe` 的容量敏感结构。
- 因为当前没有其它调度实现，该参数本身不是 hit 曲线调节旋钮。
- 若未来增加其它 schedule，hit 机制需要重新按访问顺序分析。

### `--scan-hot-objects`

代码角度：控制进入 ladder prime/probe 的 hot 对象数量 `N`。`budget_ladder_order()` 会选出前 `N` 个满足访问次数要求的 hot key，先访问它们的第 1 次访问，然后每轮 probe 再访问这些 key 的后续访问。

理论影响：

- 增大后，`estimated_hot_working_set_tokens` 通常上升，容量饱和点右移。
- 原始预期是 hot 对象数增多会让曲线颗粒度更细；本次实测只观察到陡崖右移或移出区间，没有观察到明显变平。
- 如果增大过多而容量普遍不足，低容量 hit 会下降。
- 本轮 A->B->C 显示：70 到 90 把陡崖从 0.86 推到 0.88；110 则让 0.77-0.90 全部停在约 0.533 平台。因此它是“饱和点位置”强旋钮，不再应被视为单独的“平滑斜坡”旋钮。

### `--scan-cold-objects`

代码角度：控制每轮 cold scan 取多少个 unique cold 对象。总 cold 需求为 `scan_cold_objects * scan_probe_rounds`。

理论影响：

- 增大后，每轮写入更多 cold KV，eviction 压力上升。
- 低容量 hit rate 下降，斜坡起点右移。
- 减小后，cold scan 挤出的 hot KV 更少，低容量 hit rate 被抬高。
- 如果低容量 hit 长时间处在过低平台，说明 cold 压力偏强，可降到 18-20 或减小 cold prompt 长度。

### `--scan-probe-rounds`

代码角度：控制 `cold scan + hot probe` 的重复轮数。轮数越多，ladder 前缀越长，同时要求 hot 对象至少有 `probe_rounds + 1` 次访问。

理论影响：

- 增大后，统计样本更多，hit rate 更稳定。
- 每轮都会施加 cold scan 压力，能够重复测试缓存是否真正保留 hot KV。
- 对曲线位置和斜率的影响通常弱于 `scan-hot-objects` 和 `scan-cold-objects`。
- 不建议作为首选调参旋钮，通常保持与 `hot_repeats - 1` 匹配。

## 5. ShareGPT 对象参数

### `--sharegpt-path`

代码角度：控制 ShareGPT 源 trace 路径。脚本从该路径读取会话，优先取长会话作为候选，再由 `--random-seed` 控制 shuffle。

理论影响：

- 路径本身不直接调节 hit，但数据源的文本长度分布会影响 hot/cold prompt token。
- 换成更长的数据源会扩大 working set 或 cold scan token，曲线可能右移或低容量 hit 下降。
- 换成更短的数据源则相反，并可能削弱压力 trace 的容量敏感性。

### `--sharegpt-groups`

代码角度：控制从 ShareGPT 源数据中构造多少个候选对象。脚本先按最长会话读取，再 shuffle，然后按 `--hot-ratio` 拆成 hot/cold。

理论影响：

- 在 ladder 所需 hot/cold 数量已经满足时，主要影响 ladder 后追加的尾部请求，对核心曲线影响较小。
- 当 `scan-hot-objects` 或 cold 需求增大后，它决定对象供给是否足够。
- 若报 hot/cold 对象不足，应提高该值或调整 hot/cold 比例。

### `--hot-ratio`

代码角度：控制 ShareGPT candidates 中多少比例被标记为 hot。hot 对象会生成 `hot_repeats` 次访问，cold 对象只生成一次访问。

理论影响：

- 如果 `scan-hot-objects` 不变且 hot 供给充足，核心 ladder 基本不变。
- 增大可提供更多 hot key，支撑更大的 `scan-hot-objects`。
- 增大也会减少 ShareGPT cold 对象供给，可能导致 cold scan unique 对象不足。

### `--hot-repeats`

代码角度：控制每个 ShareGPT hot 对象生成多少次访问。`budget_ladder` 需要 1 次 prime 加每轮 1 次 probe，所以必须大于等于 `scan_probe_rounds + 1`。

理论影响：

- 从不合法值提高到合法值会让 ladder 可生成。
- 超过最低要求的访问不会进入核心 ladder 前缀，而是追加到尾部。
- 继续增大通常会提高尾部重复访问的 hit，但不显著改变核心容量-hit 斜坡。

### `--hot-context-words`

代码角度：控制 ShareGPT hot prompt 截取的上下文词数上限。该值越大，单个 hot prompt 的 `prompt_est_tokens` 通常越大。

理论影响：

- 增大会扩大 hot working set，使完整保留全部 hot KV 所需容量上升，饱和点右移。
- 如果对象数量不变，只增大单对象大小，曲线更可能整体移动，而不是自然变平。
- 值过大时，每个对象变成较粗颗粒，容量跨过阈值时仍可能出现台阶。
- 本轮 B->D、C->E 证实了这一点：hot working set 从 52.6k 到 65.1k、从 67.2k 到 85.3k 后，曲线没有平滑，只是高平台消失在当前 0.90 以内的预算区间外。

### `--cold-context-words`

代码角度：控制 ShareGPT cold prompt 截取的上下文词数上限。该值越大，每个 cold 对象写入的 KV 越多。

理论影响：

- 增大会提高每轮 cold scan token，增强 eviction 压力。
- 低容量 hit rate 降低，斜坡起点右移。
- 减小能抬高低容量 hit，适合修正“低容量平台过低”的问题。
- 与 `scan-cold-objects` 作用类似，但它是单对象大小维度的连续细调。

### `--min-context-words`

代码角度：过滤 ShareGPT seed text 过短的对象。候选文本少于该词数会被跳过。

理论影响：

- 提高后，prompt 长度更稳定，过短对象减少，hit 曲线方差可能降低。
- 提高也会减少可用候选数量，可能导致供给不足。
- 对 hit 的方向不固定，因为它同时影响 hot 和 cold 的平均大小。

### `--random-seed`

代码角度：控制 ShareGPT candidates shuffle 的随机性。

理论影响：

- 相同数据和参数下保证可复现。
- 不同 seed 会选择不同 token 长度和内容的对象，造成二阶波动。
- 大方向通常弱于对象数、上下文长度和 cold scan 强度。

### `--prompt-prefix-mode`

代码角度：当前只写入 metadata 字段；prompt 构造始终使用 `object_marker`，没有根据该参数改变实际 prompt 文本。

理论影响：

- 按当前实现，它不改变 prefix-cache hit。
- 若未来实现不同 prefix 模式，它才会影响 cold/hot 前缀复用结构。

## 6. RAG/HotpotQA 参数

### `--rag-requests`

代码角度：控制 RAG 侧总 access 数。脚本通过 `exact_hot_cold_counts()` 按 `rag_hot_ratio` 和 `rag_hot_repeats` 推导 RAG hot 对象数和 cold 对象数。

理论影响：

- 主要影响 RAG 对象供给和 ladder 后的尾部请求。
- 当 RAG 对象进入 ladder 时，增加该值可补充 hot/cold key 数量。
- RAG prompt 通常比 ShareGPT 长上下文短，因此对 hot working set 的放大作用通常弱于 ShareGPT 参数。

### `--rag-hot-ratio`

代码角度：控制 RAG access 预算中 hot 对象的比例。

理论影响：

- 增大能提供更多 RAG hot key，帮助支撑更大的 `scan-hot-objects`。
- 增大也会减少 RAG cold key，可能导致 unique cold 供给不足。
- 若 ladder 主要由 ShareGPT 对象满足，它对核心曲线影响有限。

### `--rag-hot-repeats`

代码角度：控制每个 RAG hot 对象生成多少次访问，必须满足 `rag_hot_repeats >= scan_probe_rounds + 1`。

理论影响：

- 与 `--hot-repeats` 类似，主要影响合法性和尾部重复访问。
- 超出 ladder 需求的重复访问通常不改变核心 hit 斜坡。

### `--rag-hot-chunk-words`

代码角度：控制 RAG hot chunk 的词数上限。hot prompt 中每个 chunk 会按该上限截断。

理论影响：

- 增大会放大 RAG hot prompt，扩大 RAG hot working set。
- 若 RAG hot 对象被选入 ladder，饱和点会右移，斜坡宽度可能增加。
- 由于 RAG chunk 通常较短，它更适合作为补充旋钮，不是 H1 首选。

### `--rag-cold-chunk-words`

代码角度：控制 RAG cold chunk 的词数上限。

理论影响：

- 增大会放大 RAG cold prompt，提高 cold scan token。
- 低容量 hit rate 下降，斜坡起点右移。
- 减小会降低 cold 压力。

### `--rag-hot-chunks-per-query`

代码角度：控制每个 RAG hot prompt 拼接多少个 chunk。

理论影响：

- 增大会近似按 chunk 数放大 RAG hot prompt。
- 如果这些对象进入 ladder，会扩大 hot working set。
- 同时也可能改变 prompt token 方差。

### `--rag-cold-chunks-per-query`

代码角度：控制每个 RAG cold prompt 拼接多少个 chunk。

理论影响：

- 增大会提高每个 cold RAG 对象 token 数，增强 eviction 压力。
- 低容量 hit rate 下降，斜坡起点右移。

### `--hotpotqa-max-examples`

代码角度：控制最多读取多少 HotpotQA example 来构造 chunk group。

理论影响：

- 供给充足时，对核心 hit 曲线影响有限。
- 供给不足时，提高它可以支撑更多 RAG hot/cold 对象。

### `--hotpotqa-path` 和 `--download-hotpotqa`

代码角度：控制 HotpotQA 数据来源和是否允许下载。

理论影响：

- 对同一份数据没有直接 hit 影响。
- 换数据版本会改变 chunk 文本和 token 分布，从而间接改变曲线。

## 7. 输出和运行参数

### `--out`

代码角度：控制 JSONL trace 输出路径，同时生成 `.summary.json`。

理论影响：不改变 trace 内容，只影响后续实验读取哪份文件。

### `--keep-other-traces`

代码角度：默认会清理 `data/edgekv_traces` 下旧 trace；设置该参数后保留旧文件。

理论影响：不改变当前 trace hit，但能避免或保留多份实验输入。

### `--timeout-s`

代码角度：控制 HotpotQA 加载/下载相关超时。

理论影响：只影响数据准备是否成功，不直接影响 hit。

## 8. H1 调参建议

当前问题不是陡崖位置不对，而是陡崖只会平移：A/B 是低平台到高平台的跳变，C/D/E 是单一平台。继续只增大 `scan-hot-objects` 或 `hot-context-words`，大概率只是把跳变继续右移到 0.90 之外。

下一轮建议改成先验证“容量层次”而不是继续放大：

1. 保留 B 组附近作为基准：`scan-hot-objects=90`、`hot-context-words=500`。B 的低端 hit 已接近 0.47，且高平台仍在 0.88/0.90 内，是最适合继续拆台阶的起点。
2. 在 B 基础上细扫 budget：例如 `0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89`。先确认陡崖是 vLLM block/容量粒度导致的窄过渡，还是粗网格漏掉了斜坡。
3. 如果细扫仍是一步跳变，优先改访问结构：把单轮 `24` 个 cold scan 改成多级 cold scan，例如每轮内交替 `cold 小批 -> probe 子集 -> cold 小批 -> probe 子集`，让 eviction 压力分段发生。
4. 尝试降低每个 hot 对象的 token 粒度而不是继续增大：例如保持 `scan-hot-objects=90-110`，把 `hot-context-words` 降到 `350-450`，同时用更多对象补足 working set。目标是让每个对象跨 block 的大小更细。
5. 再调 cold 压力：`scan-cold-objects=20/22/24/26` 或 `cold-context-words=700/800/900`。它主要调低端 floor 和跳变位置，不应单独期待它消除台阶。
6. `sharegpt-groups`、`hot-ratio`、`rag-requests` 只按供给需求调整；供给足够后，不把它们当核心 hit 曲线旋钮。

调节目标是让容量增加时，probe 阶段 hot 对象从“少量命中”逐步过渡到“多数命中”。这样 LPE 才能通过保留高 score 对象，在多个中低容量 budget 上相对 LRU/LFU 产生稳定 hit rate 优势，并进一步反映到 p95 TTFT。

## 9. 参数影响速查

| 参数 | 代码作用 | hit rate 影响 |
|---|---|---|
| `scan-hot-objects` | ladder hot 对象数 | 实测主要右移饱和点；70->90 陡崖 0.86->0.88，110 推出当前区间 |
| `scan-cold-objects` | 每轮 unique cold 数 | 增大 cold 压力，低容量 hit 降低，主要移动 floor 和陡崖位置 |
| `scan-probe-rounds` | cold/probe 重复轮数 | 提高统计稳定性和重复压力，非首选调参旋钮 |
| `reuse-schedule` | 选择访问重排策略 | 当前固定为 budget ladder，不是调参旋钮 |
| `sharegpt-path` | ShareGPT 源数据 | 通过文本长度分布间接影响 token 压力 |
| `hot-context-words` | ShareGPT hot prompt 大小 | 实测主要扩大容量需求；500->700 未变平，容易把高平台推出区间 |
| `cold-context-words` | ShareGPT cold prompt 大小 | 增大 eviction 压力，降低低容量 hit |
| `hot-ratio` | ShareGPT hot 对象比例 | 主要影响 hot/cold 供给，间接支撑 ladder 规模 |
| `hot-repeats` | ShareGPT hot 访问次数 | 满足 ladder 合法性；超出部分主要影响尾部 hit |
| `sharegpt-groups` | ShareGPT 对象池大小 | 主要是供给约束 |
| `rag-requests` | RAG access 总数 | 主要是 RAG 供给和尾部规模 |
| `rag-hot-chunk-words` | RAG hot chunk 大小 | 增大 RAG hot working set，作用通常弱于 ShareGPT |
| `rag-cold-chunk-words` | RAG cold chunk 大小 | 增大 cold 压力 |
| `rag-hot/cold-chunks-per-query` | 每个 RAG prompt 的 chunk 数 | 放大对应 hot/cold prompt token |
| `min-context-words` | ShareGPT 最短文本过滤 | 稳定长度分布，但可能减少供给 |
| `random-seed` | ShareGPT 选择顺序 | 引入二阶 token 分布波动 |
| `prompt-prefix-mode` | 当前仅 metadata | 当前实现不影响 hit |
