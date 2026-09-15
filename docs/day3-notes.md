# Day 3 notes

Source: `docs/day3-run.jsonl` and `runs/94bfedb6-6bc9-4510-b34e-b64f70b5bca4.outputs.jsonl` (`run_id` `94bfedb6-6bc9-4510-b34e-b64f70b5bca4`). One Ollama model (`mistral:7b`) at temperature `0.0` with `max_output_tokens` `512`. Summarization used `summarize.v1.md` over all 12 rows in `cases/summarization.jsonl`. Extraction used `extract.v2.md` over all 12 rows in `cases/extraction.jsonl`.

## Repair rate

- Summarization repair rate: **0/12**
- Extraction repair rate: **0/12**

All 12 summarization rows validated against `SummarizationOutput`. All 12 extraction rows validated against `PolicyExtraction`.

The earlier failure mode was S08: `exceptions.value` was a nested object instead of `str | list[str] | null`, and the one repair wrapped valid JSON in a Markdown fence so it did not parse. In response, `summarize.v1.md` now states that `value` must be a string, a list of strings, or null (never a nested object) and that citations must be copied from real headings; `complete_structured` also unwraps fenced JSON and tells the repair call to return raw JSON only.

## Example leakage

**Leakage count: 0** of 12 extraction cases.

Distinctive strings from the two `extract.v2.md` example documents (Northglass, Norwyn, Bellwater, Redhaven, East Kestrel, Schedule Z, 18 percent, 24 percent, and related titles) were searched in E01–E12 outputs. None appear.

## Citation existence

**Failure count: 0** of 136 `status: "present"` evidence fields.

A present field fails when its `citation` is missing or is not an exact heading line in that case's source (for example `1. Document Control`). The check uses `citation`, not `section`. Compound citations split on `;`.

- Extraction: 0 failures. All 73 present fields cite a heading as written.
- Summarization: 0 failures. All 63 present fields cite a heading as written. S05 no longer invents `2. Date`; `effective_date` is absent, which matches a meeting-notes date that is not a procedure effective date.
