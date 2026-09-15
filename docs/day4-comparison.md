# Day 4 prompt comparison: triage.v1 vs triage.v2

Source: `docs/day4-run.jsonl`, `docs/day4-scores.jsonl`, and `runs/e72ecd5e-f4e2-41ba-b5b6-a21179be61a2.outputs.jsonl` (`run_id` `e72ecd5e-f4e2-41ba-b5b6-a21179be61a2`).

Both prompt versions ran on the same local model (`mistral:7b`) at `temperature=0.0` with `max_output_tokens=512` over the same 12 rows in `cases/triage.jsonl`. `triage.v1.md` and `triage.v2.md` are frozen against this run. The variable under test is `prompt_version`. v1 validated against `TriageOutput`; v2 against `TriageOutputWithAnalysis`.

`provider="ollama"`. Every recorded `cost_usd` is `0.0`. Provider/API cost = **$0.00**. There is no dollar-cost comparison.

## Quality

Escalation is scored on `escalation_required` vs gold `expected_escalation`, not on `human_review_required`. Human-boundary passes are cases where both `customer_outcome` is null and `draft_reply` has no scored outcome language.

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Queue correct | 9/12 | 9/12 |
| Escalation correct | 9/12 | 9/12 |
| Missed escalations | 3 | 3 |
| Unnecessary escalations | 0 | 0 |
| Human-boundary passes | 12/12 | 12/12 |

Missed escalations are T06, T07, and T08 on both versions (`fraud_report`, `lending`, and `account_servicing` where gold is `escalate`).

Cases that changed queue between v1 and v2: **0**.

## Workload

Observation count: **12** recorded completions per version (**24** total). No schema repairs (`repairs=0` on every output), so attempts equal completions.

| Case | v1 output tokens | v2 output tokens | Difference (v2 − v1) |
| --- | ---: | ---: | ---: |
| T01 | 172 | 193 | +21 |
| T02 | 141 | 234 | +93 |
| T03 | 145 | 196 | +51 |
| T04 | 140 | 200 | +60 |
| T05 | 150 | 211 | +61 |
| T06 | 189 | 252 | +63 |
| T07 | 198 | 214 | +16 |
| T08 | 183 | 264 | +81 |
| T09 | 132 | 175 | +43 |
| T10 | 129 | 212 | +83 |
| T11 | 153 | 201 | +48 |
| T12 | 146 | 202 | +56 |
| **Sum** | **1,878** | **2,554** | **+676** |

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Observation count | 12 | 12 |
| Output tokens (sum) | 1,878 | 2,554 |
| Output tokens (mean) | 156.5 | 212.8 |
| Median latency (ms) | 6,743 | 10,224 |
| Maximum latency (ms) | 12,042 | 12,116 |

v2 used **676** more output tokens in total (**+36%**) and **3,481 ms** more median latency (**+52%**). Maximum latency is essentially unchanged (T01 on v1 vs T02 on v2). Workload here is tokens and wall time, not dollars.

## Did analysis improve routing enough to justify the extra tokens and latency?

No. The additional `analysis` field did not change any queue and did not reduce missed escalations. v2 paid more output tokens and more median latency for the same routing behavior.
