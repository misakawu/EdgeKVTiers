# TODO: 生成 budget-sensitive 的 H0/H1 pressure trace

## 目标

生成一种 trace，使真实 vLLM GPU prefix-cache hit 满足：

- hit 随 `gpu_memory_utilization` / 可用 KV blocks 增加而增加。
- 最低可启动预算不能也接近满 hit；低预算应有明显 eviction/miss。
- 在固定 `batch_size=8`、未饱和条件下，至少一个工作点落入 `hit_rate=0.5-0.85`。

`uf750` / `uf900` 的结论说明：只改 `reuse_key/session_id/rag_reuse_key` 没用；vLLM prefix cache 看 prompt token 前缀。要控制真实 hit，必须控制 prompt 开头的 token 共享关系和复用距离。

## 从 uf750/uf900 得到的判据

1. metadata-only 负控无效：只改 reuse metadata 时，`uf250/uf500/uf750/uf900` hit 仍约 `0.952`。
2. 在 prompt 开头加入 per-request unique marker 后，hit 被拉低：
   - `uf750`: hit `0.686856`，qwait/p95 `0.000038`，有效窗口。
   - `uf900`: hit `0.566836`，qwait/p95 `0.000038`，有效窗口。
3. 因此当前 trace 的高 hit 很大概率来自“所有请求共享相同开头模板/公共前缀块”，而不是 COP 字段里的对象复用。
4. 想让 hit 随预算增加，trace 需要让 reuse distance 分布跨过不同预算容量：
   - 一部分 hot prefix 的两次访问之间插入的 distinct KV blocks 要大于最低预算容量，所以低预算会丢。
   - 同一部分 hot prefix 的复用距离又要小于中/高预算容量，所以更高预算能保住。
   - 不能让所有请求都共享同一个全局开头模板，否则低预算也会命中公共 blocks，掩盖预算效应。

## 推荐 trace 形状

### A. 不要让所有请求共享同一个 prompt 开头

当前 `build_sharegpt_prompt()` 以固定模板开头：

```text
Use the following ShareGPT-derived context as the fixed working memory.
Context:
...
```

RAG prompt 也有固定 `Retrieved context:` 结构。vLLM 的 block 级 prefix cache 会优先命中这些公共开头。建议新增一个模式，例如 `--prefix-family-mode object`：

- hot 对象：同一对象的所有 repeat 使用完全相同的对象前缀。
- cold 对象：每个对象开头加入唯一 family marker，避免与其他 cold/hot 共享开头。
- 不要把唯一 marker 放在 prompt 末尾；必须放在开头几个 token 内。

建议 prompt 结构：

```text
[TRACE_OBJECT sharegpt_hot_0007]
<对象真实 context 或 chunk text>

Task: ...
Assistant:
```

cold：

```text
[TRACE_OBJECT sharegpt_cold_0123]
<cold context>
...
```

这样 hit 主要来自“同一对象 repeat”，而不是全局模板。

### B. 用 scan-resistant 复用距离制造预算敏感性

目标序列不是简单均匀插入 hot copies，而是按 hot 对象分 phase：

```text
# prime: 先装入一批 hot objects
H0_0, H1_0, ..., H15_0

# cold scan: 插入足够多 unique/cold prefix，制造 cache pressure
C0, C1, ..., Ck

# probe: 再访问同一批 hot objects
H0_1, H1_1, ..., H15_1

# 重复多轮
C{k+1}, ..., C{2k}
H0_2, H1_2, ..., H15_2
```

其中 `C` 必须是真正 unique prompt prefix，不是只改 metadata。`k * avg_cold_tokens` 应该处在不同预算之间：

- 如果最低可启动预算约只能容纳 `B_low` tokens，则让 cold scan tokens `> B_low`。
- 如果中/高预算可容纳 `B_mid/B_high` tokens，则让 cold scan tokens `< B_mid` 或至少让一部分 hot 对象 `< B_mid`。

这样低预算 miss，中/高预算 hit，hit 才会随预算上升。

### C. 使用对象大小分层，而不是全都很短或全都很长

建议保留三类对象：

| 类别 | 目的 | 生成方式 |
| --- | --- | --- |
| hot-small | 产生可保留复用 | 128-256 words，repeat 3-5 次 |
| hot-medium | 制造预算边界 | 300-600 words，repeat 2-4 次 |
| cold-large | 施加 eviction pressure | 600-1000 words，repeat 1 次，unique prefix |

如果 hot 太小且 repeats 太近，最低预算也能保住，hit 会饱和。如果 cold 太大或 scan 太长，所有预算都 miss。需要让 hot-medium 的 reuse distance 横跨 low/mid/high 容量。

## `scripts/optimize_h0_pressure_trace.py` 修改建议

### 1. 新增 prompt prefix 模式

增加参数：

```text
--prompt-prefix-mode shared_template|object_marker|unique_cold
```

推荐默认改成 `object_marker` 或新增实验用 preset：

- `shared_template`: 当前行为，保留兼容。
- `object_marker`: 每个对象 prompt 第一行放稳定 object marker；同一 hot 对象 repeat 共享 marker。
- `unique_cold`: cold 对象第一行唯一 marker，hot 对象按 object marker 复用。

实现点：

- 修改 `build_sharegpt_prompt(context, task_index)`，传入 `object_id/object_type`。
- RAG 的 `build_rag_query_session()` 也要把 marker 放到 `turns[0].user` 最开头，或者把 retrieved context 前的第一行改成对象 marker。

### 2. 修改 hot/cold 排列，不只 evenly_insert_hot_copies

当前 `evenly_insert_hot_copies()` 容易让 repeat 过近或无法明确控制 reuse distance。建议新增：

```text
--reuse-schedule even|scan_resistant|budget_ladder
```

`budget_ladder` 推荐逻辑：

1. 选 `N_hot` 个 hot objects，先各访问一次。
2. 插入 `scan_cold_objects` 个 cold unique objects。
3. 再访问这批 hot objects。
4. 重复 `probe_rounds` 轮。
5. 剩余对象追加到尾部。

已有 `scan_resistant_prefix()` 接近这个结构，但要确保 cold/hot 的 prompt 开头真实不同，不能共享全局模板。

### 3. 显式输出 trace 诊断信息

生成 trace 后打印或写 sidecar JSON：

- `total_requests`
- `unique_prompt_prefix_families`
- `hot_objects`, `hot_accesses`, `hot_repeats`
- `cold_objects`, `cold_accesses`
- `avg_hot_tokens`, `avg_cold_tokens`
- `scan_rounds`
- `estimated_cold_scan_tokens_per_round`
- `estimated_hot_working_set_tokens`
- `expected_behavior`: low/mid/high 预算下应 hit 或 miss 的对象层判断

这能在跑 vLLM 前先判断 trace 是否可能 budget-sensitive。

### 4. 推荐起始 preset

先实现一个保守 preset，用来替代当前高 hit pressure trace：

```bash
python3 scripts/optimize_h0_pressure_trace.py \
  --out data/edgekv_traces/sharegpt_hotpotqa_session.jsonl \
  --sharegpt-groups 96 \
  --hot-ratio 0.20 \
  --hot-repeats 4 \
  --hot-context-words 300 \
  --cold-context-words 800 \
  --rag-requests 128 \
  --rag-hot-ratio 0.20 \
  --rag-hot-repeats 4 \
  --rag-hot-chunk-words 120 \
  --rag-cold-chunk-words 320 \
  --rag-hot-chunks-per-query 1 \
  --rag-cold-chunks-per-query 3 \
  --scan-hot-objects 16 \
  --scan-cold-objects 24 \
  --scan-probe-rounds 3 \
  --prompt-prefix-mode object_marker \
  --reuse-schedule budget_ladder
```

如果最低预算 hit 仍高：

- 增大 `scan-cold-objects` 或 `cold-context-words`。
- 减少 `hot-repeats`，避免 repeat 太密。
- 确认 cold/hot prompt 第一行没有共享同一模板。

如果所有预算 hit 都低：

- 减小 `scan-cold-objects` 或 `cold-context-words`。
- 减小 hot object 大小，或增加 hot repeat 轮数。
- 让一部分 hot-small 的 reuse distance 小于 low budget，保留基础 hit。

## 直接可用的会话复用方式

不改脚本也可以先按下面方式手工构造 JSONL：

1. 每个 hot object 生成 3-4 个访问，所有访问的 prompt 第一行完全一致：`[TRACE_OBJECT HOT_0007]`。
2. 每个 cold object 只访问一次，prompt 第一行唯一：`[TRACE_OBJECT COLD_0123]`。
3. 顺序采用：`hot prime -> cold scan -> hot probe -> cold scan -> hot probe`。
4. cold scan 的总 token 数要大于最低预算 KV 容量，但不要大到超过最高预算太多。
5. 所有 prompt 不要共享同一个固定开头；公共说明文本放到 object marker 后面。

这个结构的预期结果是：低预算在 probe 时丢掉部分 hot prefix，中/高预算保留更多 hot prefix，因此 hit 随显存预算增加。
