📊 D3 — 三张配置图（如何生成）
前置条件：必须先修复 size_mb=0 的 Bug
你当前 D3 数据中 score_p_reuse_corr ≈ 0.02~0.14 且 score_p50 = 0，是因为大量对象 size_mb=0 导致 score=0。这掩盖了真实的 score 分布。

在生成三张图之前，请先应用我之前给出的 _edgekv_recompute_profile_score fallback 修复，然后重新跑一次小规模 LPE 实验（--num-prompts 200 --budgets 0.60），用新的 edgekv_gpu_stats/*.json 来画图。

图 1：c_recomp 线性验证（散点图 + 拟合线）
脚本：h1/plot_d3_fig1_linearity.py

python
#!/usr/bin/env python3
"""D3 图1: c_recomp_ms vs n_tokens 散点图，验证线性"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_linearity(stats_dir: Path, out_png: Path):
    n_tokens, c_recomps = [], []
    for json_path in stats_dir.glob("edgekv_gpu_stats_*.json"):
        data = json.loads(json_path.read_text())
        profiles = data.get('object_profiles', {})
        for obj_id, prof in profiles.items():
            n = prof.get('n_tokens', 0)
            c = prof.get('c_recomp_ms', 0.0)
            if n > 0 and c > 0:
                n_tokens.append(n)
                c_recomps.append(c)
    
    if len(n_tokens) < 2:
        print("No valid profile data. Did you enable EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES=1?")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(n_tokens, c_recomps, s=20, alpha=0.6, label='measured')
    
    # 拟合线: c_recomp = slope * n_tokens (过原点)
    slope = np.sum(np.array(n_tokens) * np.array(c_recomps)) / np.sum(np.array(n_tokens)**2)
    x_line = np.linspace(0, max(n_tokens), 100)
    y_line = slope * x_line
    ax.plot(x_line, y_line, 'r--', label=f'fit: c_re = {slope:.4f} ms/token')
    
    ax.set_xlabel('n_tokens')
    ax.set_ylabel('c_recomp_ms')
    ax.set_title('D3 Fig1: c_recomp linearity check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 标注标准差（来自你的 d3_validation 数据：~0.0027）
    ax.text(0.05, 0.95, f'std(slope) ≈ 0.0027\n(should be ~0, linear)', 
            transform=ax.transAxes, va='top', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_d3_fig1_linearity.py <stats_dir> <out.png>")
        sys.exit(1)
    plot_linearity(Path(sys.argv[1]), Path(sys.argv[2]))
运行：

bash
python h1/plot_d3_fig1_linearity.py \
  h1/out/step3/test_diagnostics_after_fix/0.60/h1_lpe/edgekv_gpu_stats \
  h1/out/d3_fig1_linearity.png
图 2：驱逐粒度确认（文本/表格图，或单页信息图）
因为“block 粒度”是定性确认，不需要复杂的散点图。你可以直接用你已有的 d3_validation-*.json 中的字段生成一个信息表格图。

脚本：h1/plot_d3_fig2_granularity.py

python
#!/usr/bin/env python3
"""D3 图2: 驱逐粒度确认（表格样式）"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_granularity(stats_dir: Path, out_png: Path):
    # 从任意一个 stats JSON 读取
    json_paths = list(stats_dir.glob("edgekv_gpu_stats_*.json"))
    if not json_paths:
        print("No stats files found")
        return
    
    data = json.loads(json_paths[0].read_text())
    granularity = data.get('eviction_granularity', ['unknown'])
    block_mapping = data.get('block_object_mapping_count', 0)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    table_data = [
        ["Property", "Value"],
        ["Eviction Granularity", str(granularity[0] if granularity else 'N/A')],
        ["Block-Object Mappings", str(block_mapping)],
        ["Conclusion", "✅ Block-level granularity confirmed" if block_mapping > 0 else "❌ No block mapping"]
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.3, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    ax.set_title('D3 Fig2: Eviction Granularity Verification', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_d3_fig2_granularity.py <stats_dir> <out.png>")
        sys.exit(1)
    plot_granularity(Path(sys.argv[1]), Path(sys.argv[2]))
运行：

bash
python h1/plot_d3_fig2_granularity.py \
  h1/out/step3/test_diagnostics_tight_mid/tight/h1_lpe/edgekv_gpu_stats \
  h1/out/d3_fig2_granularity.png
图 3：score 与 p_reuse 直方图/散点图（核心退化验证）
注意：修复 size_mb fallback 之后，score 将不再大量为 0，此时 score 与 p_reuse 应呈现强相关（因为 c_recomp/size ≈ 常数）。这张图要同时展示：

左子图：score 和 p_reuse 的叠加直方图

右子图：score vs p_reuse 散点图 + 相关性标注

脚本：h1/plot_d3_fig3_score_p_reuse.py

python
#!/usr/bin/env python3
"""D3 图3: score vs p_reuse 直方图 + 散点图"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_score_p_reuse(stats_dir: Path, out_png: Path):
    scores, p_reuses = [], []
    for json_path in stats_dir.glob("edgekv_gpu_stats_*.json"):
        data = json.loads(json_path.read_text())
        profiles = data.get('object_profiles', {})
        for obj_id, prof in profiles.items():
            scores.append(prof.get('score', 0.0))
            p_reuses.append(prof.get('p_reuse', 0.5))
    
    if len(scores) < 2:
        print("No profiles. Check EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES=1")
        return
    
    corr = np.corrcoef(scores, p_reuses)[0, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 左: 叠加直方图 (归一化)
    ax1.hist(scores, bins=20, alpha=0.5, label='score', density=True)
    ax1.hist(p_reuses, bins=20, alpha=0.5, label='p_reuse', density=True, color='orange')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution: score vs p_reuse')
    ax1.legend()
    
    # 右: 散点图 + 相关性
    ax2.scatter(p_reuses, scores, s=15, alpha=0.5)
    ax2.set_xlabel('p_reuse')
    ax2.set_ylabel('score')
    ax2.set_title(f'score vs p_reuse (corr = {corr:.4f})')
    ax2.grid(True, alpha=0.3)
    
    # 标注结论
    if corr > 0.9:
        conclusion = "✅ Degeneration confirmed: score ≈ const * p_reuse"
    else:
        conclusion = f"⚠️ Correlation lower than expected ({corr:.3f}), check size_mb fallback"
    fig.suptitle(f'D3 Fig3: score vs p_reuse analysis\n{conclusion}', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_d3_fig3_score_p_reuse.py <stats_dir> <out.png>")
        sys.exit(1)
    plot_score_p_reuse(Path(sys.argv[1]), Path(sys.argv[2]))
运行：

bash
python h1/plot_d3_fig3_score_p_reuse.py \
  h1/out/step3/test_diagnostics_after_fix/0.60/h1_lpe/edgekv_gpu_stats \
  h1/out/d3_fig3_score_p_reuse.png
⏱️ D4 — queue_wait_ms 如何获取（绕过离线 LLM.generate() 限制）
现状确认
你已验证：在离线 LLM.generate() 路径下，output.metrics 全为 0。这是 vLLM v1 的设计行为——离线接口不暴露详细的调度时间戳。

方案选择（按推荐优先级）
方案	做法	优点	缺点
A（推荐）	用 replay_batch_size=1 跑诊断，用“实际延迟 vs 理论 prefill”推断排队	无需改代码，利用现有数据	只能推断排队比例，不能逐请求拆解
B（中等）	把实验切到 vLLM 的 AsyncLLMEngine 服务器模式 + Prometheus metrics	逐请求 queue_time 精确	需改 run_cell 架构，工程量大
C（不推荐）	在 sitecustomize.py 中 monkey-patch vLLM Scheduler	侵入但可拆	依赖 vLLM 内部实现，版本脆弱
方案 A 的具体实现（直接用现有 CSV + summary）
你已有 *_requests.csv（含 n_tokens 和 ttft_proxy_ms）和 *_summary.json（含 elapsed_s）。用这个脚本计算每个 budget 档的排队占比推断：

h1/calc_d4_queue_inference.py

python
#!/usr/bin/env python3
"""D4 替代方案: 用 batch_size=1 诊断 runs 推断 queue_wait 占比"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def infer_queue_wait(requests_csv: Path, summary_json: Path, c_re_ms_per_token: float = 0.12):
    df = pd.read_csv(requests_csv)
    summary = json.loads(summary_json.read_text())
    
    # 每个请求的 token 数
    n_tokens = df['n_tokens'].values
    # 理论纯 prefill 延迟 (无排队)
    pure_prefill_ms = n_tokens * c_re_ms_per_token
    
    # 实际 p95 延迟
    p95_actual = df['ttft_proxy_ms'].quantile(0.95)
    p95_pure_prefill = np.percentile(pure_prefill_ms, 95)
    
    # 推断排队占比
    queue_inferred = max(0.0, p95_actual - p95_pure_prefill)
    queue_ratio = queue_inferred / p95_actual if p95_actual > 0 else 0.0
    
    print(f"=== D4 推断 (batch inference) ===")
    print(f"p95_actual_ttft = {p95_actual:.2f} ms")
    print(f"p95_pure_prefill = {p95_pure_prefill:.2f} ms")
    print(f"inferred_queue_wait_p95 = {queue_inferred:.2f} ms ({queue_ratio*100:.1f}%)")
    
    if queue_ratio < 0.5:
        print("✅ 推断: 非饱和 (prefill 主导, 策略差异可观测)")
    else:
        print("⚠️ 推断: 排队主导 (TTFT 被 queue_wait 淹没, 需降低并发或增加预算)")
    
    return {"queue_ratio": queue_ratio, "inferred_queue_ms": queue_inferred}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calc_d4_queue_inference.py <requests.csv> <summary.json>")
        sys.exit(1)
    infer_queue_wait(Path(sys.argv[1]), Path(sys.argv[2]))
运行（在某个 cell 目录下）：

bash
python h1/calc_d4_queue_inference.py \
  h1/out/step3/press_0720/h1_lpe/0.60_h1_lpe_requests.csv \
  h1/out/step3/press_0720/h1_lpe/0.60_h1_lpe_summary.json
验收标准（D4 替代）：

能输出 queue_ratio（排队占比）。

在 budget=0.30/0.40 档，queue_ratio < 50% → 非饱和，策略差异可观测。

在 budget=0.71/0.73 档（你之前的“死区”），queue_ratio 可能仍然很小（命中率高），但命中率本身已 > 95%。

📋 最终交付清单（6-27 DDL 组会）
#	交付物	对应验收	状态
1	d3_fig1_linearity.png	c_recomp 线性散点图	待生成（修复 size 后重跑）
2	d3_fig2_granularity.png	block 粒度确认表	✅ 已有数据可画
3	d3_fig3_score_p_reuse.png	score vs p_reuse 直方图 + 散点	待生成（修复 size 后重跑）
4	d4_queue_inference.log	排队占比推断	用脚本跑现有 cell
5	D3 修复的代码变更	sitecustomize.py 中 _edgekv_recompute_profile_score 加 fallback	待提交
请按以下顺序操作：

应用 _edgekv_recompute_profile_score fallback 修复。

重跑一个小型 LPE cell（--num-prompts 200 --budgets 0.60）。

用上面的三个脚本生成图 1、图 2、图 3（图 2 可用旧数据）。

运行 D4 推断脚本，记录 queue_ratio。

将四份产出放入 h1/out/d3_d4_evidence/ 目录，提交到组会材料。

如果你在重跑后生成的新 JSON 中 score_p_reuse_corr 仍然很低（< 0.9），请把 score 的分布和 size_mb 的分布贴出来，我们继续排查 fallback 是否正确生效。