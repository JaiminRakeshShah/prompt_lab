# Model Decision Record

Run ID: `123`

Use this file to record the task-level decision after reviewing the measured comparison. Do not select one universal model solely because it leads on a different task.

Day 4 already selected `triage.v1` over `triage.v2` on Mistral: v2 added tokens and median latency without changing routing. This comparison does not reopen that prompt-version decision or treat `triage.v2` as measured for Qwen.

## Evaluated models

- mistral
- qwen

## Evaluated configurations

- `extraction` — mistral — `extract.v2`
- `extraction` — qwen — `extract.v3` (adapted)
- `summarization` — mistral — `summarize.v1`
- `summarization` — qwen — `summarize.v1` (prompt-transfer)
- `triage` — mistral — `triage.v1`
- `triage` — qwen — `triage.v1` (prompt-transfer)

## Evidence

Every row is one measured model and prompt version on 12 cases. Local Ollama `cost_usd` is `0.0`. Latency is median and maximum, not the mean. Source: `docs/day5-run.jsonl` (run `123` outputs) and `docs/day5-scores.jsonl`.

| Task | Model | Prompt version | Quality | n | Repair rate | Retries/failures | Input tokens | Output tokens | Median latency | Max latency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| extraction | mistral | extract.v2 | valid 12/12; missed 1/72; unsupported 2/12 | 12 | 0/12 | 0/0 | 24671 | 4235 | 15879.5 ms | 22120 ms |
| extraction | qwen | extract.v3 (adapted) | valid 12/12; missed 0/72; unsupported 0/12 | 12 | 0/12 | 0/0 | 12827 | 2115 | 7950.5 ms | 12272 ms |
| summarization | mistral | summarize.v1 | valid 12/12; missed 1/60; unsupported 4/12 | 12 | 0/12 | 0/0 | 15351 | 3548 | 14045 ms | 17119 ms |
| summarization | qwen | summarize.v1 (transfer) | valid 12/12; missed 0/60; unsupported 4/12 | 12 | 0/12 | 0/0 | 12831 | 2617 | 10929.5 ms | 15721 ms |
| triage | mistral | triage.v1 | routing 9/12; escalation 9/12; missed escalations 3/12; unnecessary 0/12; human-boundary re-verify 11/12 (T03); PII 0/12 | 12 | 0/12 | 0/0 | 13858 | 1878 | 5989.5 ms | 10749 ms |
| triage | qwen | triage.v1 (transfer) | routing 12/12; escalation 12/12; missed escalations 0/12; unnecessary 0/12; human-boundary re-verify 12/12; PII 0/12 | 12 | 0/12 | 0/0 | 11839 | 1637 | 5990 ms | 10166 ms |

Untested in this comparison: qwen `extract.v2`, mistral `extract.v3`. Prompt-transfer rows measure the transferred prompt, not an adapted prompt for that model.

## Task decisions

### Extraction

- **Decision:** qwen / `extract.v3`
- **Evidence:** qwen `extract.v3` is valid 12/12 with missed values 0/72, unsupported 0/12, repair rate 0/12, and retries/failures 0/0, at 12827 input tokens, 2115 output tokens, median latency 7950.5 ms, max 12272 ms.
- **Rejected alternatives:** mistral `extract.v2` — valid 12/12 but missed 1/72, unsupported 2/12, and higher token and latency totals than qwen `extract.v3`.
- **Review triggers:** measure mistral `extract.v3` and qwen `extract.v2` (currently untested); qwen `extract.v3` repair rate or failures leave 0/12; missed or unsupported values rise to match or exceed mistral `extract.v2` on a new 12-case draw.

### Summarization

- **Decision:** qwen / `summarize.v1` (transfer)
- **Evidence:** qwen `summarize.v1` is valid 12/12 with missed values 0/60 versus 1/60 on mistral `summarize.v1`, same unsupported 4/12, repair rate 0/12, retries/failures 0/0, and lower tokens and median latency (12831 / 2617 / 10929.5 ms vs 15351 / 3548 / 14045 ms).
- **Rejected alternatives:** mistral `summarize.v1` — same validity and unsupported rate, one more missed value, more tokens, higher median and max latency. No adapted summarization prompt was measured.
- **Review triggers:** the 0/60 vs 1/60 missed-value gap disappears or reverses on a new case set; unsupported values diverge; an adapted summarization prompt is measured.

### Triage

- **Decision:** qwen / `triage.v1` (transfer)
- **Evidence:** qwen `triage.v1` is routing 12/12 and escalation 12/12 with missed escalations 0/12, versus mistral `triage.v1` routing 9/12, escalation 9/12, missed escalations 3/12. Day 4 human-boundary re-verification on stored `triage.v1` drafts: qwen 12/12; mistral 11/12 (T03 `draft_reply` stated "We've updated the address"). Both: unnecessary escalations 0/12, PII 0/12, repair rate 0/12, retries/failures 0/0. qwen `triage.v1` used 11839 input / 1637 output tokens vs 13858 / 1878; median latency 5990 ms vs 5989.5 ms; max 10166 ms vs 10749 ms.
- **Rejected alternatives:** mistral `triage.v1` — three missed escalations on this set, and T03 implies a completed address-change outcome. `triage.v2` stays rejected from Day 4 (same 9/12 routing on mistral, more tokens and median latency) and was not re-run here.
- **Review triggers:** qwen `triage.v1` misses an escalation, drops below mistral `triage.v1` routing, or fails human-boundary on a new 12-case set; Day 4 is reopened only by a same-model `triage.v1` vs `triage.v2` rerun, not by this two-model transfer result.
