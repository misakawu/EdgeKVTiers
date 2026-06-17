# H0/H1/H2/H4/H5/H3 预实验 TODO

来源：`论文规划/00_王祯祥_论文工作规划.md` 的 §0.5 预实验规划。本文只做 TODO 清单，不删除原文约束；每条 TODO 均附规划原文。

## 全局前置 TODO

- [ ] 建环境并锁版本：安装/确认 `vllm`、`datasets`、`lmcache`、`kivi`、`h2o`，输出 `env.lock`。

  原文：`装环境锁版本（vllm/datasets/lmcache/kivi/h2o）env.lock；拉 ShareGPT 200 会话转统一回放 JSON；写含 quality 维度的解析仿真器 sim.py...`

- [ ] 下载/准备模型 `Qwen2.5-7B-Instruct`，并准备端侧 4-bit 版本。

  原文：`模型：Qwen2.5-7B-Instruct；端侧用其 4-bit 版。`

- [ ] 下载/准备 ShareGPT 多轮对话数据，取约 200 条多轮会话。

  原文：`ShareGPT 多轮对话：HuggingFace datasets 上 anon8231489123/ShareGPT_Vicuna_unfiltered（或 RyokoAI/ShareGPT52K），load_dataset 后每条是 conversations:[{from,value}...]，取 from=="human" 的轮做用户输入、按对话切成会话；取前 ~200 条多轮会话即可。`

- [ ] 准备 RAG chunk 复用 trace：HotpotQA 或自建 5-10 篇文档，切 chunk 后构造跨 query 复用访问序列。

  原文：`RAG chunk 复用 trace：用 HotpotQA / 自建 5–10 篇文档，把每篇切成若干 chunk，造一份「不同 query 复用相同 chunk」的访问序列。`

- [ ] 定义统一回放 JSON 格式，并保证同一份回放文件喂给所有策略。

  原文：`统一回放格式（一个 JSON 文件）... 回放器按 session_id 顺序、turn 顺序把「累计前缀」打给引擎；同一份回放文件喂给所有策略，保证可比。`

- [ ] 建立 §6.5 trace JSONL 输出规范，所有预实验事件统一落盘。

  原文：`所有事件落进六章 §6.5 的 trace JSONL。`

## H0：trace 回放、缓存事件与指标闭环

- [ ] 启动 vLLM 默认 KV 管理基线，并开启 prefix caching。

  原文：`基线：vLLM 默认 KV 管理（开 prefix caching）。`

- [ ] 写 trace 回放器，按固定 session/turn 顺序把累计前缀请求发送给 vLLM。

  原文：`写一个 trace 回放器，按时间/顺序把请求打给 vLLM。`

- [ ] 采集每个缓存事件：命中、未命中、驱逐，并写入 §6.5 trace。

  原文：`采集每个缓存事件（命中/未命中/驱逐）为一条 §6.5 trace。`

- [ ] 实现 H0 指标脚本：p95 TTFT、cache hit rate、GPU memory peak、每 token 重算代价 `c_re`。

  原文：`指标脚本：p95 TTFT、cache hit rate、GPU memory peak、每 token 重算代价 c_re。`

- [ ] 用一次重算实验标定 `c_re`。

  原文：`算 p95 TTFT、hit rate、显存峰值；用一次重算实测标定 c_re。`

- [ ] 确认指标随并发/显存预算变化合理。

  原文：`确认指标随并发/显存预算变化合理（显存越小命中率越低）。`

- [ ] H0 trace 每条记录至少包含 `event`、`hit`、`n_tokens`、`size_mb`、`t_policy_ms`；summary CSV 包含 p95 TTFT 和显存峰值。

  原文：`测什么 + trace 字段：event、hit、n_tokens、size_mb、t_policy_ms（H0 可为 0）；外加 p95 TTFT/显存峰值（CSV）。`

- [ ] 跑通 H0 通过条件：trace 能回放、三类指标可算、事件落盘。

  原文：`通过 = trace 能回放、三类指标可算、事件落盘；不通过 = 回放或指标采集卡死。`

## H1：生命周期策略 vs LRU/LFU

- [ ] 在 H0 回放器中实现可插拔驱逐策略：LRU、LFU、score/LPE。

  原文：`在 H0 的回放器里实现可插拔驱逐策略（LRU/LFU/score）。`

- [ ] 实现 COP 规则版画像：`p_reuse` 用 LRU-K/频次衰减，`c_recomp = c_re * n`。

  原文：`实现 COP 规则版画像（p_reuse 用 LRU-K/频次衰减，c_recomp=c_re·n）。`

- [ ] 实现 LPE score：`score = p_reuse * c_recomp / size`，按单位显存收益最低者驱逐。

  原文：`验证"按 score=p_reuse·c_recomp/size 驱逐"是否真比 LRU/LFU/vLLM默认 更省 p95 TTFT。`

- [ ] 使用 Qwen2.5-7B 和 ShareGPT 多会话 + RAG chunk 复用混合负载。

  原文：`模型：Qwen2.5-7B；负载：ShareGPT 多会话 + RAG chunk 复用混合。`

- [ ] 在紧/中/松三档显存预算与不同并发会话数下，跑 LRU、LFU、vLLM 默认、LPE 四种策略。

  原文：`对照：① LRU；② LFU；③ vLLM 默认；④ LPE（score 驱逐 + COP 画像）。自变量：显存预算 M_budget（紧/中/松三档）、并发会话数。`

- [ ] 记录 p95 TTFT、hit rate、显存峰值。

  原文：`四策略在三档显存预算下各跑同一 trace。记录 p95 TTFT、hit rate、显存峰值。`

- [ ] 输出“显存预算 vs p95 TTFT”四条线。

  原文：`画"显存预算 vs p95 TTFT"四条线，比较同预算下的差距。`

- [ ] H1 trace 增加 `p_reuse`、`score`、`lpe_action`、`hit` 字段。

  原文：`测什么 + trace 字段：p_reuse、score、lpe_action、hit；p95 TTFT/显存峰值（CSV）。`

- [ ] H1 通过条件：多数显存档 LPE 比 LRU/LFU 降 p95 TTFT ≥20%，且显存不超预算。

  原文：`通过 = 多数显存档位下 LPE 比 LRU/LFU 降 p95 TTFT ≥20% 且显存不超预算；不通过 = 收益不稳或仅个别档位。`

## H2：restore-vs-recompute 避免 I/O 反噬

- [ ] 从 H1 混合 trace 中筛出“被驱逐后又复用”的对象集。

  原文：`模型：Qwen2.5-7B；负载：H1 的混合 trace 中被驱逐后又复用的对象。`

- [ ] 实现三种策略：always-restore、always-recompute、RRS。

  原文：`对照：① 总是恢复（always-restore）；② 总是重算（always-recompute）；③ RRS（定理 1 阈值）。`

- [ ] 实现可配置 offload 带宽 `BW`，支持模拟 PCIe/SSD/网络；必要时真实 offload 到 CPU/SSD 校准。

  原文：`实现可配置 BW 的卸载/恢复模拟（或真实 offload 到 CPU/SSD 测 BW）。`

- [ ] 实现 RRS 判据：比较 `c_restore = mu_kv * n / BW + d_deser` 与 `c_recomp = c_re * n`。

  原文：`RRS 用 c_restore=mu_kv n/BW+d_deser vs c_recomp=c_re n 判定。`

- [ ] 对每个复用且曾被驱逐对象，三策略各算/测一次代价。

  原文：`对每个"复用且曾被驱逐"的对象，三策略各算/测一次代价。`

- [ ] 扫多档 `BW`，记录 restore latency、recompute ratio、p95 latency。

  原文：`扫多档 BW，记录 restore latency、recompute ratio、p95 latency。`

- [ ] 输出“带宽 vs p95”三条线，检查 RRS 是否不高于两固定策略下包络。

  原文：`画"带宽 vs p95"三条线，找 RRS 是否始终 ≤ 两固定策略的下包络。`

- [ ] H2 trace 增加 `rrs_action`、`c_restore_ms`、`c_recomp_ms`、`bw_gbps`、`hit` 字段。

  原文：`测什么 + trace 字段：rrs_action、c_restore_ms、c_recomp_ms、bw_gbps、hit；p95 latency（CSV）。`

- [ ] H2 通过条件：RRS 在多数带宽档不劣于两固定策略，低带宽下显著优于 always-restore。

  原文：`通过 = RRS 在多数带宽档不劣于两固定策略、低带宽下显著优于 always-restore；不通过 = I/O 成本不可控或 RRS 判据失效。`

## H4：quality 维度真带来正收益

- [ ] 下载/接入 KIVI/H2O kernel，用于 int8/int4/sparse-k 等精度机制。

  原文：`用现成 KIVI/H2O kernel 实测 quality-size-restore 三元组（H4）`；`控制变量：同 trace、同模型、同后端、同 baseline 量化机制（都用 KIVI kernel）。`

- [ ] 准备 LongBench 任务，用于度量 task 质量损失。

  原文：`负载：ShareGPT 多会话 + RAG chunk 复用 + LongBench 任务（用于度量 task 质量损失）。`

- [ ] 对每个精度实测 `(size_factor, quality_loss_per_token, restore_factor)` 三元组，包含 perplexity 与 LongBench 任务。

  原文：`用 KIVI/H2O 实测 (size_factor, quality_loss_per_token, restore_factor) 三元组（对每精度，跑 perplexity 与 LongBench 任务）。`

- [ ] 实现 TMS 跨精度迁移逻辑：先 best-fit-decreasing 静态版，再加在线迁移。

  原文：`仿真器里实现 TMS 跨精度迁移逻辑（先 best-fit-decreasing 静态版，再加在线迁移）。`

- [ ] 实现四个对照：RAGCache(PGDSF 仿真，二元 cache)、static-int4、LPE-score、TMS-tiered。

  原文：`对照：① RAGCache(PGDSF 仿真，二元 cache)；② static-int4（全用 KIVI int4，不调度）；③ LPE-score（仅 lifecycle，不 quality）；④ TMS-tiered（quality × lifecycle 联合）。`

- [ ] 扫 9 个 `(M_budget, epsilon)` 组合：质量预算紧/中/松 × 显存预算紧/中/松。

  原文：`自变量：质量预算 epsilon（紧/中/松三档）、显存预算 M_budget（紧/中/松三档）。`；`9 个 (M_budget, epsilon) 组合下跑四对照。`

- [ ] 报告 p95 TTFT、累积 quality loss、显存峰值、降级次数。

  原文：`报告：p95 TTFT、累积 quality loss、显存峰值、降级次数。`

- [ ] 输出“质量预算 epsilon vs p95 TTFT”四条线。

  原文：`画"质量预算 epsilon vs p95 TTFT"四条线，关键看 TMS 与次优 baseline 的差距是否 ≥25%。`

- [ ] H4 trace 增加 `q(o)`、`qloss_per_event`、`tms_action`、`tier_migrations_total` 字段。

  原文：`测什么 + trace 字段：q(o)、qloss_per_event、tms_action（hold/downgrade/upgrade/evict）、tier_migrations_total；p95 TTFT/累积 quality loss/显存峰值（CSV）。`

- [ ] H4 通过条件：多数 `(M_budget, epsilon)` 组合下 TMS 比次优 baseline 降 p95 TTFT ≥25%，且 quality loss ≤ epsilon。

  原文：`通过 = 多数 (M_budget, epsilon) 组合下 TMS 比次优 baseline 降 p95 TTFT ≥25%，且 quality loss 严格 ≤ epsilon；不通过 = quality 维度收益 <15% → 退方案 B1。`

## H5：质量-带宽相变面经验可观测

- [ ] 使用 H4 仿真器和实测三元组作为 H5 输入。

  原文：`用 H4 的仿真器 + 实测三元组。`

- [ ] 扫 35 个网格点：`BW in [0.5,1,2,4,8,16,32]` GB/s × `epsilon in [0.5,1,2,4,8]`。

  原文：`自变量：BW∈[0.5, 1, 2, 4, 8, 16, 32] GB/s × epsilon∈[0.5, 1, 2, 4, 8]（per-token bits 等价）= 35 个网格点。`

- [ ] 实现 per-object brute-force 离线最优函数，求每个对象的 `q_star`。

  原文：`仿真器加 brute-force 求 q* 的离线最优函数（per object）。`

- [ ] 在每个网格点跑 200 个代表性对象，记录每对象 `q_star`。

  原文：`在 35 个 (BW, epsilon) 网格点跑 200 个代表性对象，记每对象的 q*。`

- [ ] 实现定理 1 解析预测，并用 Kendall-tau 衡量 `q_star` 与预测一致性。

  原文：`用 Kendall-tau 测 q* 与定理 1 解析预测的一致性。`

- [ ] 输出热力图 `(BW, epsilon) -> q_star`，检查块状 phase transition。

  原文：`画热力图 (BW, epsilon) -> q*，验证块状结构。`

- [ ] H5 trace 增加 `q_star_offline`、`q_star_theorem_pred`、`grid_point` 字段。

  原文：`测什么 + trace 字段：q_star_offline、q_star_theorem_pred、grid_point；Kendall-tau 一致性、热力图。`

- [ ] H5 通过条件：热力图明显块状相变 + Kendall-tau ≥ 0.8。

  原文：`通过 = 热力图明显呈现块状相变 + Kendall-tau ≥ 0.8；不通过 = q* 在网格上随机分布或单调 → 定理 1 退化为不等式描述，不当 first-class theorem 写。`

## H3：策略层不改引擎也能落地

- [ ] 准备服务器 1×GPU 和端侧受限显存 GPU/Jetson 测试环境。

  原文：`设备：服务器 1×GPU + 端侧（受限显存 GPU/Jetson）。`

- [ ] 使用 H1 的混合 trace 作为 H3 负载。

  原文：`负载：H1 的混合 trace。`

- [ ] 确认 LMCache/vLLM 暴露 admission、eviction、offload 钩子。

  原文：`确认 LMCache/vLLM 暴露的 admission/eviction/offload 钩子。`

- [ ] 将 COP/LPE/RRS 作为策略层挂到 LMCache/vLLM 外围，先接 eviction，再接 offload/restore。

  原文：`把 COP/LPE/RRS 挂上去（先只接 eviction，再接 offload/restore）。`

- [ ] 实现策略层对照：引擎默认 vs 策略层接入（COP/LPE/RRS）。

  原文：`对照：① 引擎默认；② 策略层接入（COP/LPE/RRS）。`

- [ ] 单独计时每次策略决策开销 `T_policy`。

  原文：`单独计时每次策略决策开销 T_policy。`

- [ ] 比较接/不接策略层的 p95 TTFT、hit rate、显存峰值。

  原文：`比较接/不接策略层的 p95 TTFT、hit rate、显存峰值。`

- [ ] 服务器和端侧各跑一遍，确认趋势一致。

  原文：`端侧与服务器各做一遍，确认趋势一致。`

- [ ] H3 trace 增加 `t_policy_ms`、`lpe_action`、`rrs_action`、`hit` 字段。

  原文：`测什么 + trace 字段：t_policy_ms、lpe_action、rrs_action、hit；p95 TTFT/显存峰值（CSV）。`

- [ ] H3 通过条件：策略层稳定收益、`T_policy` 可忽略、每次决策可由 trace 解释、端侧趋势一致。

  原文：`通过 = 策略层带来稳定收益、T_policy 可忽略、每次决策可由 trace 解释、端侧趋势一致；不通过 = 接入困难或策略开销吃掉收益。`

## 交付物 TODO

- [ ] W1 交付：解析仿真器 + H1/H2 两条趋势图。

  原文：`W1 | 第 0 步 + H1/H2 趋势 | 解析仿真器（含 quality 维度）跑通：LRU/LFU/score/tiered 四策略 + RRS 判据，出 H1/H2 初步趋势（用示意量化三元组）| 仿真器 + 两条趋势图。`

- [ ] W2 交付：H0 回放闭环 + `c_re/mu_kv/d_deser` 真标定值 + KIVI/H2O 真三元组。

  原文：`W2 | H0 + 量化三元组标定 | 接 vLLM 默认（开 prefix caching）+ ShareGPT trace 回放，标定 c_re/mu_kv/d_deser；用 KIVI/H2O 实测 (size_factor, qloss, restore_factor) 替换示意值 | 回放闭环 + 真标定值 + 真三元组。`

- [ ] W3 交付：用真标定值跑 H1 三档显存预算四策略对比，输出显存-p95 曲线。

  原文：`W3 | H1 | 用真标定值在三档显存预算下跑四策略对比，画显存-p95 曲线 | H1 结论图。`
