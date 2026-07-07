# H0 Heterogeneous Probe Conclusion

- Effective window budgets: 0.25, 0.35, 0.45, 0.60, 0.70
- Heterogeneous LPE wins in effective window: 5
- Verdict: LPE score needs heterogeneous objects to become useful; the next step should move toward H4 quality/quantization evidence.

## Homogeneous Degeneration

| budget | LRU p95_cost | LPE p95_cost | LPE-LRU |
| ---: | ---: | ---: | ---: |
| 0.25 | 14.877060 | 14.960308 | 0.083247 |
| 0.35 | 14.689199 | 14.787109 | 0.097911 |
| 0.45 | 14.483095 | 14.554274 | 0.071178 |
| 0.60 | 14.094985 | 14.156611 | 0.061626 |
| 0.70 | 13.575929 | 13.582454 | 0.006526 |
| 0.80 | 13.310086 | 13.272252 | -0.037834 |
| 0.90 | 13.054810 | 13.054810 | 0.000000 |

## Heterogeneous Effective Window

| budget | LRU hit_rate | LRU p95_cost | LPE p95_cost | LRU-LPE |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.615556 | 12.354928 | 11.508375 | 0.846553 |
| 0.35 | 0.695370 | 11.675097 | 11.172170 | 0.502927 |
| 0.45 | 0.760926 | 11.387728 | 11.077018 | 0.310710 |
| 0.60 | 0.823519 | 11.051629 | 10.804274 | 0.247355 |
| 0.70 | 0.849630 | 10.772050 | 10.747947 | 0.024103 |
