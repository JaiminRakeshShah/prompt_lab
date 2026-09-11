# Day 2 model comparison

Source: `docs/day2-run.jsonl` (`run_id` `4feb6c0f-074a-4fec-a3cf-3cb4f880e74a`). Both models used `provider="ollama"` with `cost_usd=0.0`. There is no provider/API charge to compare.

| Metric | Mistral | Qwen |
| --- | ---: | ---: |
| Cases | 12 | 12 |
| Successful (`error_type` null) | 12 | 10 |
| Truncated (`stop_reason=length`) | 0 | 2 (S06, S10) |
| Input tokens (sum) | 2,787 | 2,427 |
| Output tokens (sum) | 1,142 | 4,806 |
| Latency median (ms) | 3,728 | 17,882 |
| Latency max (ms) | 8,659 | 23,995 |

Qwen used about four times as many output tokens and about five times the median latency as Mistral on the same prompt and cases. The 512-token output ceiling stopped two Qwen answers; Mistral finished every case under that cap. Workload here is tokens and wall time, not dollars.
