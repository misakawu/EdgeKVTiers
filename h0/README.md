# H0 trace replay

## ShareGPT trace

The H0 ShareGPT run uses this local trace path:

```text
/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

The path is also encoded in `configs/sharegpt_server_edge.json` and used as the
default ShareGPT path by `run_h0.py`.

Run command:

```bash
python3 h0/run_h0.py --config h0/configs/sharegpt_server_edge.json --out out/h0_sharegpt_server_edge
```

Latest validation output:

- `passed`: true
- `devices`: `server_sharegpt`, `edge_sharegpt`
- `objects`: 120
- `total_requests`: 800
- `events`: 1600
- `token_ref`: 116173

Key result interpretation:

- Both device profiles replay the same ShareGPT trace and share the same
  `token_ref`, so server/edge numbers are comparable.
- Both runs respect the quality budget: `epsilon_norm = 0.0002` and
  `epsilon_ok = true`.
- Memory pressure is active in both profiles: server peaks at `1199.82 MB`
  under a `1200 MB` budget, while edge peaks at `519.87 MB` under a `520 MB`
  budget.
- Edge has similar total hit rate but shifts more hits to offloaded objects:
  server offload hit rate is `0.558442`, edge offload hit rate is `0.644156`.
- Edge latency is higher because lower bandwidth and higher deserialization cost
  make restore/recompute paths more expensive: p95 TTFT is `152.763 ms` on edge
  versus `135.93 ms` on server.
