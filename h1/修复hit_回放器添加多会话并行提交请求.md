### 1. 改造目标与影响范围

- **目标**：将请求顺序从“按原始文件顺序连续处理”改为“**按会话轮询（Round-Robin）**”，即每次轮流从不同会话中各取一个未处理的请求。
- **影响模块**：回放器中的 **请求排序阶段**（`order_trace_for_batches` 函数），**不影响**批处理拆分逻辑（`replay_batches`）和 vLLM 引擎执行逻辑。

---

### 2. 核心算法思路（伪代码）

**输入**：原始扁平请求列表 `trace`（每个元素包含 `session_id`、`turn_index` 等信息）。  
**输出**：按会话交替重排后的新请求列表 `ordered_trace`。

```
function order_trace_round_robin(trace):
    // 1. 按 session_id 分组
    groups = empty_map(session_id -> list_of_requests)
    orphan_requests = []   // 无 session_id 的请求（如单轮 RAG）

    for req in trace:
        if req.has_field("session_id") and req.session_id 非空:
            groups[req.session_id].append(req)   // 每个会话内部按原始 turn_index 升序
        else:
            orphan_requests.append(req)

    // 2. 初始化轮询状态
    pointers = map(session_id -> 0)   // 每个会话当前的轮次指针
    active_sessions = groups.keys()   // 集合

    ordered = []

    // 3. 轮询循环
    while active_sessions 非空:
        for sid in active_sessions 的当前快照（或遍历前先复制列表）:
            idx = pointers[sid]
            if idx < groups[sid].length:
                ordered.append(groups[sid][idx])
                pointers[sid] = idx + 1
            else:
                // 该会话已取完，标记为待移除

        // 清理已完成的会话（避免在迭代中修改集合）
        active_sessions = 过滤掉 pointers[sid] >= groups[sid].length 的会话

    // 4. 追加孤立请求
    ordered.extend(orphan_requests)

    return ordered
```

---

### 3. 与现有批处理逻辑的衔接

重排后的 `ordered_trace` 会送入 `replay_batches` 函数进行切分。现有 `replay_batches` 已包含“同一会话的多轮强制拆分”逻辑（`must_split`），因此：

- 即使轮询导致连续两个请求来自同一会话（例如某会话只剩最后两轮），`replay_batches` 也会自动将它们拆到不同批次。
- **建议保留该拆分逻辑**，因为真实并发场景下，同一会话的下一轮请求本就不会与前一轮同时出现在同一批次中（它们有先后依赖）。

---

### 4. 为什么这能解决“显存压力低”的问题？

- **低并发（batch_size=1）情况下**：无论顺序如何，每次只有一个请求在引擎中，显存占用必然低。这并非排序能解决的问题。
- **高并发（batch_size=8或更大）情况下**：
  - 原始顺序可能导致一个批次内包含多个请求，但可能集中在少数几个会话的早期轮次。
  - 轮询顺序确保每个批次（尤其是多个批次同时排队时）包含**来自不同会话**的请求，vLLM 调度器会并行处理它们，使得多个独立的 KV Cache 上下文同时驻留显存，从而显著提高 `GPU KV cache usage`。
- **结论**：轮询排序本身 **不直接** 提升显存占用，但它与 `replay_batch_size` 配合，能让高并发条件下的显存压力更真实、更均匀，避免因请求顺序聚集导致的缓存冷热不均。

---

### 5. 预期对命中率的影响

- 轮询会**拉长**同一会话相邻轮次之间的时间间隔（中间插入了其他会话的请求）。
- 这可能导致部分中间轮次的缓存块被其他会话的缓存块**驱逐**，从而略微降低会话内部历史前缀的命中率。
- 但这也更接近真实线上场景：多用户并发时，每个用户的上下文都可能被挤出缓存，这正是缓存替换策略发挥作用的地方。因此，这种排序更适合评估 LRU/LFU/LPE 等策略的**公平性和鲁棒性**。

---

### 6. 如何集成到实验框架

- 在已有的 `--batch-order` 参数中新增一个选项，例如 `round_robin`。
- 当该选项启用时，调用上面的轮询排序函数；否则走原有的 `original` 或 `length_bucket` 路径。
- 命令行示例（伪）：
  ```
  python run_h1_vllm0110_real.py \
    --batch-order round_robin \
    --replay-batch-size 8 \
    --max-num-batched-tokens 32768
  ```

---

### 7. 潜在边界情况处理

- **会话轮数不均**：若某些会话很长（如 20 轮），而多数会话很短（2 轮），长会话会独自撑到最后阶段。这符合真实场景，无需特殊处理。
- **超大量会话**：分组和指针维护开销与会话数成正比，通常 trace 含数百会话，完全可接受。
- **无会话 ID 的请求**（如独立的 RAG 查询）：将它们视为“孤儿”，追加到轮询序列末尾，避免干扰轮询逻辑。

---

### 8. 总结

通过上述伪代码逻辑，你可以：

1. 将请求顺序从“集中式”改为“交替式”。
2. 结合增大 `replay_batch_size`，使多个会话的请求同时在 GPU 中排队，产生有效的显存压力。
3. 更真实地模拟线上多用户并发场景，为缓存替换策略提供更有说服力的对比数据。