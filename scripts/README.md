# Trace 参数对 budget-hit 曲线的影响

本文记录 `optimize_h0_pressure_trace.py` 中各生成参数对 LRU prefix-cache hit 曲线的影响，用于继续把
`data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` 的 budget-hit 曲线从台阶调成平缓斜坡。

## 当前观测

旧 trace 的结构指标：

- `estimated_hot_working_set_tokens = 9624`
- `estimated_cold_scan_tokens_per_round ~= 18k-20k`
- LRU hit：`0.77 ~= 0.51`，`0.80 ~= 0.90`，之后平台

v1 参数：

```bash
--sharegpt-groups 200 --hot-ratio 0.30 --hot-repeats 4
--hot-context-words 500 --cold-context-words 800
--scan-hot-objects 70 --scan-cold-objects 24 --scan-probe-rounds 3
--rag-requests 160 --rag-hot-ratio 0.20 --rag-hot-repeats 4
```

生成后的结构指标：

- `estimated_hot_working_set_tokens = 39700`
- `estimated_cold_scan_tokens_per_round = [18281, 17052, 17130]`

本轮 partial LRU 结果：

| budget | LRU hit |
|---:|---:|
| 0.77 | 0.379475 |
| 0.80 | 0.379928 |
| 0.83 | 0.378571 |
| 0.86 | 0.958264 |
| 0.88 | 0.958264 |

结论：v1 已经把陡崖位置从 `0.77-0.80` 推到 `0.83-0.86`，但仍是台阶，不是平缓斜坡。
这说明当前调整主要改变了“饱和点/陡崖位置”，还没有足够改善“斜坡颗粒度”。

## 机制模型

`budget_ladder` 的访问序列是：

```text
prime N 个 hot 对象
repeat probe_rounds 次:
  扫一批 cold 对象
  复测全部 hot 对象
```

LRU hit 主要由两个量决定：

- `estimated_cold_scan_tokens_per_round`：每轮 cold 扫描会挤掉多少旧 KV。
- `estimated_hot_working_set_tokens`：cold 扫描后，容量增加时能逐步保留下来的 hot 前缀总量。

粗略理解：

- cold scan 决定低 budget 的 floor。
- hot working set 决定 ramp 的宽度和饱和点。
- hot object 数量决定曲线颗粒度；数量太少时，即使 hot working set 变大，也容易还是一个大台阶。

## 参数影响

下面按脚本里的实际作用分组说明。`trace 影响` 描述生成出来的访问序列、对象数量或 prompt token 变化；`hit 预测` 描述对 LRU prefix-cache hit 曲线的预期方向。

### 调度参数

#### `--reuse-schedule`

当前唯一可选值是 `budget_ladder`。

- trace 影响：固定生成 `prime hot -> cold scan -> hot probe` 的前缀结构，后面再追加未被前缀使用的剩余请求。
- hit 预测：这是让 budget-hit 曲线对 KV 容量敏感的核心结构；没有其它调度模式可用于对比。

#### `--scan-hot-objects`

选择多少个 hot 对象进入 `budget_ladder` 的 prime/probe 主结构。

- trace 影响：prime 阶段写入更多 hot key，每轮 probe 也会复测更多 hot key；`estimated_hot_working_set_tokens` 通常随之上升，前缀请求数也上升。
- hit 预测：增大会让 hit 曲线颗粒更细、ramp 更宽，饱和点通常右移；减小会让饱和点提前，曲线更容易变成少量大台阶。
- 调参优先级最高。对象数太少时，单对象再大也容易形成台阶；对象数增加但单对象 token 太小，则曲线会更细但不一定足够宽。

#### `--scan-cold-objects`

每轮 cold scan 使用多少个 unique cold 对象。

- trace 影响：每轮 cold scan token 总量近似按对象数线性增长，要求至少存在 `scan-cold-objects * scan-probe-rounds` 个 unique cold key。
- hit 预测：增大会提高 eviction 压力，低 budget floor 降低，ramp 起点右移；减小会保留更多 hot 前缀，低 budget hit 升高。
- 如果低 budget hit 明显低于目标，应先降低 cold 压力；如果低 budget hit 明显高于目标，应提高 cold 压力。

#### `--scan-probe-rounds`

控制 `cold scan + hot probe` 重复多少轮。

- trace 影响：增加会拉长 ladder 前缀，并要求每个 hot 对象至少有 `scan-probe-rounds + 1` 次访问，同时需要更多 unique cold 对象。
- hit 预测：轮数更多会让统计更稳定，也会反复施加 eviction 压力；但它主要改变测量稳定性和 replay 成本，不是首选 hit 旋钮。
- 当前 `3` 与 `hot-repeats=4`、`rag-hot-repeats=4` 正好匹配，不建议优先调。

### ShareGPT 参数

#### `--sharegpt-path`

ShareGPT 源 trace 路径。

- trace 影响：决定 ShareGPT seed text 的来源；脚本按 `order="longest"` 读取候选，再用 seed shuffle。
- hit 预测：如果换成长度分布不同的数据，hot/cold prompt token 会整体变化，从而移动 floor、ramp 和饱和点；路径本身不是连续调参旋钮。

#### `--sharegpt-groups`

从 ShareGPT 中构造多少个对象组。

- trace 影响：增加候选对象池，按 `--hot-ratio` 切成 hot/cold；最终生成的 ShareGPT access 数为 `hot_objects * hot_repeats + cold_objects`。
- hit 预测：在 `scan-hot-objects` 和 `scan-cold-objects * scan-probe-rounds` 不变时，主要影响 ladder 之后的尾部组成，核心 hit 曲线变化有限；当对象不足时报错时，它是供给约束旋钮。

#### `--hot-ratio`

ShareGPT 对象中 hot 对象比例。

- trace 影响：增加会把更多 ShareGPT group 标成 hot，并生成重复访问；减少会增加 cold group。
- hit 预测：若 ladder 选中的 hot 数不变，核心曲线变化有限；若原本 hot key 不够，增大它能支撑更大的 `scan-hot-objects`，间接提高颗粒度并右移饱和点。
- 如果生成器报 `budget_ladder needs ... hot objects`，提高 `--hot-ratio` 或 `--sharegpt-groups`。

#### `--hot-repeats`

每个 ShareGPT hot 对象有多少次访问。

- trace 影响：必须满足 `hot-repeats >= scan-probe-rounds + 1`，因为需要 1 次 prime 加每轮 1 次 probe；超出的重复访问会留在 ladder 之后的尾部。
- hit 预测：从低于最低要求提高到合法值会让 trace 可生成；继续增大通常不改变 ladder 前缀中的核心 eviction/retention 结构，只可能提高尾部 hit。

#### `--hot-context-words`

ShareGPT hot prompt 截取的上下文词数上限。

- trace 影响：增加会放大单个 ShareGPT hot 前缀，提升 `avg_hot_prompt_est_tokens` 和 `estimated_hot_working_set_tokens`。
- hit 预测：增大会把饱和点推向更高 budget，并拓宽容量从“只能保留部分 hot”到“能保留全部 hot”的区间；减小会让饱和点左移。它更擅长移动陡崖位置，不一定能单独消除台阶。
- 本轮 v1 将 `300 -> 500` 后，配合 `scan-hot-objects 35 -> 70`，hot working set 从 `9.6k` 提到 `39.7k`，陡崖也从 `0.77-0.80` 移到 `0.83-0.86`。

#### `--cold-context-words`

ShareGPT cold prompt 截取的上下文词数上限。

- trace 影响：增加会放大单个 ShareGPT cold 对象，每轮 cold scan token 上升；减少则降低 cold scan token。
- hit 预测：增大会降低低 budget floor，并把 ramp 起点右移；减小会抬高低 budget hit。相比 `--scan-cold-objects`，它是更连续的 cold 压力细调旋钮。

#### `--min-context-words`

ShareGPT seed text 的最低词数门槛。

- trace 影响：提高会过滤掉短上下文，让候选 prompt 更长、更稳定，但可用对象减少；降低会引入更短对象。
- hit 预测：提高通常会增加平均 ShareGPT prompt token，使 hot working set 和 cold scan 压力都上升，具体 hit 方向取决于被 ladder 选中的是 hot 还是 cold；主要用于控制数据质量和避免过短 prompt。

### RAG/HotpotQA 参数

#### `--rag-requests`

HotpotQA/RAG 侧总 access 数。

- trace 影响：由 `--rag-hot-ratio` 和 `--rag-hot-repeats` 精确拆成 hot/cold 对象；增加会提供更多 RAG hot key 和 unique cold key。
- hit 预测：若 ladder 主要选中 ShareGPT 对象，影响偏供给和尾部；若 RAG 对象进入 ladder，增加可补充颗粒度，但 RAG hot prompt 通常较短，对扩大 hot working set 的作用弱于 ShareGPT 长上下文。

#### `--rag-hot-ratio`

RAG access 预算中的 hot 占比。

- trace 影响：增大会增加 RAG hot 对象数并减少 RAG cold 对象数；减小则相反。
- hit 预测：增大可补 hot key 供给，帮助更大的 `scan-hot-objects`；但也可能减少 cold key 供给。若 cold 不足，`budget_ladder` 会报 unique cold 不够。

#### `--rag-hot-repeats`

每个 RAG hot 对象有多少次访问。

- trace 影响：必须满足 `rag-hot-repeats >= scan-probe-rounds + 1`；超出部分进入 ladder 后面的尾部。
- hit 预测：与 `--hot-repeats` 类似，主要影响生成合法性和尾部 hit，对 ladder 核心曲线影响有限。

#### `--rag-hot-chunk-words`

RAG hot chunk 的单 chunk 词数上限。

- trace 影响：增大会放大 RAG hot prompt；`reuse_key` 也包含 chunk word 配置，因此不同配置是不同对象族。
- hit 预测：增大会增加 RAG hot working set，让饱和点右移；但本轮 RAG hot 平均 token 为 `129.16`，远小于 ShareGPT hot 的 `643.08`，所以通常不是扩大 working set 的第一优先级。

#### `--rag-cold-chunk-words`

RAG cold chunk 的单 chunk 词数上限。

- trace 影响：增大会放大 RAG cold prompt，提高 cold scan token；减小会降低 cold scan token。
- hit 预测：增大会降低低 budget floor 并右移 ramp 起点；减小会抬高低 budget hit。

#### `--rag-hot-chunks-per-query`

每个 RAG hot query 拼接多少个 chunk。

- trace 影响：增加会按 chunk 数放大 RAG hot prompt，并改变 chunk set 组成。
- hit 预测：增大会扩大 RAG hot working set，饱和点右移；如果 RAG hot 被选入 ladder，也能增加斜坡宽度。

#### `--rag-cold-chunks-per-query`

每个 RAG cold query 拼接多少个 chunk。

- trace 影响：增加会按 chunk 数放大 RAG cold prompt，提高每轮 cold scan token。
- hit 预测：增大会增强 eviction，低 budget hit 下降，ramp 起点右移；减小会降低 cold 压力。

#### `--hotpotqa-path`

HotpotQA 数据路径。

- trace 影响：决定 RAG chunk group 的来源；路径不存在且未设置 `--download-hotpotqa` 时，加载会失败。
- hit 预测：换数据会改变 chunk 长度、问题文本和对象多样性，进而改变 prompt token 分布；路径本身不是连续 hit 旋钮。

#### `--hotpotqa-max-examples`

最多读取多少个 HotpotQA example 来构造 chunk group。

- trace 影响：提高会增加可构造的 RAG 对象候选；降低可能导致 hot/cold group 不足。
- hit 预测：在对象供给充足时，对核心 hit 曲线影响有限；供给不足时，提高它可以支撑更大的 RAG hot/cold ladder 规模。

#### `--download-hotpotqa`

允许脚本在需要时下载 HotpotQA。

- trace 影响：只影响数据获取路径，不改变已存在数据的 prompt 构造逻辑。
- hit 预测：对 hit 没有直接影响；只有当下载得到的数据版本或内容不同，才会间接改变 token 分布。

#### `--timeout-s`

HotpotQA 加载/下载相关操作的超时时间。

- trace 影响：超时太短会导致数据准备失败；足够长时不改变 trace 内容。
- hit 预测：无直接影响。

### 输出、随机性和标记参数

#### `--out`

输出 JSONL trace 路径。

- trace 影响：决定写入位置，并在同目录生成 `.summary.json`。
- hit 预测：无直接影响；但 H1 replay 使用哪个文件会决定实际跑到哪份 trace。

#### `--keep-other-traces`

是否保留 `data/edgekv_traces` 下其它旧 trace 文件。

- trace 影响：默认会清理同目录旧 `.jsonl` 和 summary 文件；加上该参数后不清理。
- hit 预测：不改变当前生成 trace 的 hit，只影响实验目录中是否残留旧输入，避免误用时有间接价值。

#### `--random-seed`

ShareGPT 候选 shuffle 的随机种子。

- trace 影响：改变 ShareGPT hot/cold 对象的具体选择和顺序；在参数相同且数据相同时保证可复现。
- hit 预测：会造成二阶波动。不同 seed 可能选到 token 长度不同的对象，使 working set 或 cold scan token 小幅变化；大方向应弱于对象数和上下文长度参数。

#### `--prompt-prefix-mode`

可选 `object_marker` 或 `unique_cold`，当前脚本只把该值写入 metadata，prompt 构造始终使用对象级 marker。

- trace 影响：当前不改变 prompt 文本、`reuse_key` 或访问顺序，只改变 JSON 行里的 `prompt_prefix_mode` 字段。
- hit 预测：按当前代码没有直接影响。如果未来实现 `unique_cold` 语义，应预期 cold 前缀更难复用、cold hit 下降、eviction 压力观测更纯。

## 调参方向

本轮问题是：`0.77-0.83` 低平台，`0.86` 直接到高平台。

这表示：

- hot working set 已经足以把饱和点推高，但仍没有铺开成连续斜坡。
- cold 压力偏强，导致低端 floor 从目标 `~0.5` 降到 `~0.38`。
- 需要同时降低 cold floor 压力，并增加 hot 颗粒度。

下一轮建议：

- 先把 `--scan-cold-objects` 从 `24` 降到 `20` 或 `18`，把 `0.77` floor 拉回 `~0.5`。
- 将 `--scan-hot-objects` 继续增加到 `90-110`，同时提高 `--sharegpt-groups` 和/或 `--hot-ratio` 保证对象供给。
- 不要大幅增加 `--hot-context-words`；它更可能继续移动陡崖位置，而不是消除陡崖。
- 如果增加 hot 对象后 hot working set 超过目标太多，再轻微降低 `--hot-context-words`。

优先级：

1. 用 `scan-cold-objects` / `cold-context-words` 把 `0.77` floor 校回 `~0.5`。
2. 用 `scan-hot-objects` 增加颗粒度。
3. 用 `hot-context-words` 微调饱和点。
4. 用 `sharegpt-groups` / `hot-ratio` / `rag-requests` 解决对象供给。
