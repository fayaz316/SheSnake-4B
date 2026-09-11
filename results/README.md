# Evaluation evidence

This directory contains the privacy-safe evidence behind the SheSnake-4B exploratory
local result.

- `scorecard.json`: machine-readable protocol, scores, paired outcomes, revisions,
  and SHA-256 hashes.
- `generations/*.raw.jsonl`: exact model responses before EvalPlus sanitization.
- `generations/*.jsonl`: code extracted by EvalPlus for execution.
- `evaluations/*.json`: per-task executable test outcomes.

All reported files contain 164 HumanEval task records. HumanEval+ pass counts require
both the original and added tests to pass; this matters for `HumanEval/116`, where the
SheSnake generation passed the added tests but failed an original test and is
correctly counted as a HumanEval+ failure.

The artifacts establish the reported local comparison. They do not establish an
external leaderboard rank or an improvement over untuned Qwen.
