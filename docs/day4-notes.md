# Day 4 notes

Source: `docs/day4-run.jsonl`, `docs/day4-scores.jsonl`, and `runs/e72ecd5e-f4e2-41ba-b5b6-a21179be61a2.outputs.jsonl` (`run_id` `e72ecd5e-f4e2-41ba-b5b6-a21179be61a2`). One Ollama model (`mistral:7b`) at temperature `0.0` with `max_output_tokens` `512`. Both prompt versions used the same 12 rows in `cases/triage.jsonl`. `triage.v1.md` and `triage.v2.md` are frozen against this run. Provider/API cost is `$0.00`.

Escalation is scored on `escalation_required` vs gold `expected_escalation`.

triage.v1
queue correct: 9/12
escalation correct: 9/12
missed escalations: 3
unnecessary escalations: 0
human-boundary passes: 12/12

triage.v2
queue correct: 9/12
escalation correct: 9/12
missed escalations: 3
unnecessary escalations: 0
human-boundary passes: 12/12

changed-queue count: 0/12

token difference: v2 used 2,554 output tokens vs 1,878 for v1 (676 more; 12/12 cases used more tokens on v2)

latency comparison: median 10,224 ms (v2) vs 6,743 ms (v1); maximum 12,116 ms (v2) vs 12,042 ms (v1); 12 observations per version

The v2 `analysis` field did not earn its additional overhead. Routing was unchanged on this 12-case set: same 3 missed escalations (T06, T07, T08), no queue changes, more tokens, and higher median latency. A one-case difference in a 12-case set would not prove one prompt is universally better; this run does not even have that difference.
