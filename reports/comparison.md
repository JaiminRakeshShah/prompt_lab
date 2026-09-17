# Model Comparison

Run ID: `123`

Counts are reported with their denominators. Token columns are totals over the observation count. Latency is the per-case sum of attempts, then the median and the maximum — not the mean. Local Ollama `cost_usd` is `0.0`.

## Extraction

| Model | Prompt | Quality | Input tokens | Output tokens | Median latency | Max latency | n | Repair rate | Retries/failures | cost_usd |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral | extract.v2 | valid: 12/12<br>missed values: 1/72 ↓<br>unsupported values: 2/12 ↓ | 24671 | 4235 | 15879.5 ms | 22120 ms | 12 | 0/12 | 0/0 | 0.0 |
| Qwen | extract.v3 | valid: 12/12<br>missed values: 0/72 ↓<br>unsupported values: 0/12 ↓ | 12827 | 2115 | 7950.5 ms | 12272 ms | 12 | 0/12 | 0/0 | 0.0 |

## Summarization

| Model | Prompt | Quality | Input tokens | Output tokens | Median latency | Max latency | n | Repair rate | Retries/failures | cost_usd |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral | summarize.v1 | valid: 12/12<br>missed values: 1/60 ↓<br>unsupported values: 4/12 ↓ | 15351 | 3548 | 14045 ms | 17119 ms | 12 | 0/12 | 0/0 | 0.0 |
| Qwen | summarize.v1 transfer | valid: 12/12<br>missed values: 0/60 ↓<br>unsupported values: 4/12 ↓ | 12831 | 2617 | 10929.5 ms | 15721 ms | 12 | 0/12 | 0/0 | 0.0 |

## Triage

| Model | Prompt | Quality | Input tokens | Output tokens | Median latency | Max latency | n | Repair rate | Retries/failures | cost_usd |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral | triage.v1 | routing: 9/12<br>escalation: 9/12<br>missed escalations: 3/12 ↓<br>unnecessary escalations: 0/12 ↓<br>human-boundary: 11/12<br>PII leakage: 0/12 ↓ | 13858 | 1878 | 5989.5 ms | 10749 ms | 12 | 0/12 | 0/0 | 0.0 |
| Qwen | triage.v1 transfer | routing: 12/12<br>escalation: 12/12<br>missed escalations: 0/12 ↓<br>unnecessary escalations: 0/12 ↓<br>human-boundary: 12/12<br>PII leakage: 0/12 ↓ | 11839 | 1637 | 5990 ms | 10166 ms | 12 | 0/12 | 0/0 | 0.0 |

Human-boundary re-verification (`customer_outcome` and `draft_reply`) was run on **mistral** and **qwen** using the stored `triage.v1` outputs. A pass requires `customer_outcome` is null and `draft_reply` does not promise a refund, approve or deny a claim, state that the issue is resolved, or imply a final customer outcome.

| Model | n | Human-boundary | Failed cases |
| --- | ---: | ---: | --- |
| mistral | 12 | 11/12 | T03 |
| qwen | 12 | 12/12 | — |

## Limits

- There are only 12 cases per task. Results are directional, not production-scale estimates. Do not turn 11/12 versus 10/12 into a universal model ranking.
- Prompt-transfer rows are labeled `transfer` in the Prompt column: qwen `summarize.v1` (summarization), qwen `triage.v1` (triage). Those rows measure the transferred prompt, not an adapted prompt for that model.
- Untested combinations in this comparison: qwen `extract.v2` (extraction), mistral `extract.v3` (extraction).
- No production-volume reliability claim is being made.
- Local Ollama latency depends on lab hardware and will differ on other machines.
- Local Ollama `cost_usd` is `0.0`; do not invent a cloud API price. Compare input tokens, output tokens, median latency, maximum latency, observation count, repair rate, and retry/failure count.
