# Publication and release checklist

This document is the release path for the public GitHub repository and the Hugging
Face model page. It intentionally separates local evidence from externally accepted
leaderboard placement.

## 1. Freeze the evidence

Before publishing:

1. Treat the finished SheSnake-versus-Ministral run as exploratory local evidence.
2. Record the exact `evalplus` version, conversion revisions, decoding settings,
   output counts, and final pass@1 values.
3. Export raw generations and executable result files only after removing local paths
   and checking for credentials or personal identifiers.
4. Run the full untouched-Qwen ablation before claiming that QLoRA improved the base.
5. Run `pytest -q`, `git diff --check`, and the release privacy scan.

The current two-row result is sufficient for a narrowly qualified model comparison.
It is not sufficient evidence that fine-tuning caused the difference. Keep these two
claims separate in every announcement.

The model card must state that the result is a local reproducibility result until an
external leaderboard publishes it. Do not copy upstream vendor scores into the same
table unless the benchmark, prompt format, decoding policy, and model revision are
known to match.

## 2. GitHub repository

The repository should contain source code, documentation, the verified-data manifest,
and reproducible commands. It should not contain model weights, local caches, secrets,
or machine-specific evaluation dumps.

```bash
pytest -q
git status --short
git add README.md CONTRIBUTING.md LICENSE LICENSE-NOTICE.md docs benchmark data finetune tests results pyproject.toml env.example .gitignore release
git commit -m "Prepare reproducible SheSnake-4B release"
git push origin main
```

Review the staged file list before committing. The repository remote is expected to be
`https://github.com/abrarf316/SheSnake-4B`.

## 3. Hugging Face model repository

Create a **public** model repository with the display name **SheSnake-4B** and the
canonical lowercase identifier:

```text
fayaz-mgs/shesnake-4b
```

The first release should contain:

- `README.md` based on `release/README.md`;
- `adapters.safetensors` copied from `finetune/adapters/shesnake-4b/` and the
  privacy-safe `release/adapter_config.json`;
- the upstream-license notice;
- a link to this GitHub repository;
- exact benchmark results, limitations, and reproduction commands.

The adapter is the primary artifact. Do not upload the 4-bit base checkpoint again
unless there is a clear packaging reason; link users to both the upstream Qwen model
and the exact MLX conversion used for training.

For a standalone external leaderboard submission, additionally prepare a
Transformers-compatible Safetensors export that loads through `AutoConfig`,
`AutoModel`, and `AutoTokenizer`. An MLX-only adapter may be perfectly usable locally
but may not pass an external loader check.

## 4. Model-card language

Use:

> SheSnake-4B is a QLoRA adapter for Qwen3-4B-Instruct-2507, trained in France on
> checker-verified Python examples and evaluated with executable HumanEval+ tests.

Clarify that France is training provenance rather than French-language specialization,
and qualify every score with the exact 4-bit checkpoint and local protocol.

Do not describe it as a new foundation model or as a training-from-scratch system.
Do not claim a public rank until the relevant board has accepted and displayed the
submission.

## 5. Leaderboard submission

The local HumanEval+ result is the primary exploratory technical evidence. A public
leaderboard submission is a separate experiment and may use different tasks and
loading rules.

For the Hugging Face Open LLM Leaderboard, the model must be public, use Safetensors,
load through supported Transformers AutoClasses, and include a complete model card.
Submit the exact immutable revision and classify it as a fine-tuned model. Track the
request status and report acceptance only after the board publishes the result.

## 6. Cost

Public model repositories and the local evaluation path can be completed at no API or
GPU rental cost. Hugging Face storage is subject to its current public-storage policy;
keep the release compact and upload only useful artifacts.
