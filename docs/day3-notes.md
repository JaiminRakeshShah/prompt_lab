# Day 3 notes

Source: `docs/day3-run.jsonl` and `runs/b1017e9e-8ed4-4ef2-a31d-8f7f21bb5a8f.outputs.jsonl` (`run_id` `b1017e9e-8ed4-4ef2-a31d-8f7f21bb5a8f`). One Ollama model (`mistral:7b`) at temperature `0.0` with `max_output_tokens` `512`. Summarization used the shipped `summarize.v1.md` over all 12 rows in `cases/summarization.jsonl`. Extraction used `extract.v2.md` over all 12 rows in `cases/extraction.jsonl`.

## Repair rate

- Summarization repair rate: **0/12**
- Extraction repair rate: **0/12**

All 12 summarization rows validated against `SummarizationOutput`. All 12 extraction rows validated against `PolicyExtraction`.

The most common validation error on the restored shipped prompt was S05: extra keys (`attendees`, `discussion`) forbidden by the schema, plus absent fields missing `value`. In response, those shape rules were added to the Day 3 system prompt and `complete_structured` unwraps fenced JSON; `summarize.v1.md` was left as shipped.

## Example leakage

**Leakage count: 0** of 12 extraction cases.

Distinctive strings from the two `extract.v2.md` example documents (Northglass, Norwyn, Bellwater, Redhaven, East Kestrel, Schedule Z, 18 percent, 24 percent, and related titles) were searched in E01–E12 outputs. None appear.

## Citation existence

**Failure count: 0** of 136 `status: "present"` evidence fields.

A present field fails when its `citation` is missing or is not an exact heading line in that case's source (for example `1. Document Control`). The check uses `citation`, not `section`. Compound citations split on `;`.

- Extraction: 0 failures. All 73 present fields cite a heading as written.
- Summarization: 0 failures. All 63 present fields cite a heading as written.
