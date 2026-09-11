# SheSnake-4B

**Bonjour.**

SheSnake-4B is an exploratory QLoRA adaptation of
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
for compact Python function generation. It was trained locally in France on 104
checker-verified MBPP training examples. Training was performed in France; the model
is not presented as French-language-specialized.

Under one matched local EvalPlus 0.3.1 protocol, SheSnake-4B scored **79.3%** on
HumanEval+ versus **71.3%** for the specified 4-bit MLX conversion of
Ministral-3-3B-Instruct-2512, an advantage of **8.0 percentage points**.

## Results

| Local checkpoint | HumanEval | HumanEval+ | Passing tasks |
|---|---:|---:|---:|
| SheSnake-4B | **85.4%** | **79.3%** | 140 / 164; 130 / 164 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | 75.6% | 71.3% | 124 / 164; 117 / 164 |

The paired HumanEval+ outcomes were 109 tasks passed by both models, 21 passed only
by SheSnake, 8 passed only by Ministral, and 26 passed by neither. The net difference
is 13 tasks. These are local measurements, not an accepted leaderboard placement.

The narrow claim supported by this experiment is:

> With EvalPlus 0.3.1, one greedy completion per task, and the exact 4-bit MLX
> checkpoints documented here, SheSnake-4B scored 79.3% HumanEval+ pass@1 versus
> 71.3% for Ministral-3-3B-Instruct-2512-4bit.

It does **not** establish general superiority over Ministral, the official
full-precision checkpoint, other Mistral models, or larger systems.

## What was actually trained

- Base: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- Method: MLX QLoRA; the base weights remained frozen
- Trainable parameters: approximately 3.67M of 4.02B (0.091%)
- Data: 104 MBPP `sanitized/train` examples; 16 held out for validation
- Optimization: 200 steps, batch size 1, gradient accumulation 4, learning rate
  `1e-5`, 8 adapted layers, rank 8, seed 0
- Final recorded validation loss: 0.691; this is a training diagnostic, not a
  benchmark score

Every admitted reference solution compiled and passed its supplied assertions in a
Python isolated-mode subprocess. The verifier is a quality check, not a security
sandbox.

## Evaluation protocol

- Benchmark: EvalPlus 0.3.1 HumanEval+, all 164 tasks
- Sampling: pass@1, one completion, greedy decoding, temperature 0
- Generation cap: 768 new tokens
- Controls: shared EvalPlus prompt builder, output sanitizer, and executable evaluator
- Model-specific components: each checkpoint's own tokenizer, chat template, and MLX
  server implementation
- Completion: 164 outputs and 164 evaluation records for each reported model; no
  missing generations

The community checkpoint revisions used were:

- Qwen MLX conversion: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
- Ministral MLX conversion: `a962dcb09eee4169c890e544c9eb938f1113fdee`

Aggregate evidence is in [`results/`](results/), and the full interpretation is in
[`docs/TECHNICAL-REPORT.md`](docs/TECHNICAL-REPORT.md).

## Evaluation integrity and limits

This result is legitimate under the recorded protocol, with important limits:

- No HumanEval or HumanEval+ solutions were used as supervised training targets.
- MBPP prompts were screened for high literal similarity to HumanEval+ prompts at a
  normalized similarity threshold of 0.82; zero examples were removed. This is an
  overlap screen, not proof of semantic decontamination.
- Public benchmark exposure during upstream pretraining cannot be ruled out for
  either base model.
- The first 10 HumanEval+ tasks were used as a systems gate. Ministral scored 8/10,
  untuned Qwen 9/10, and SheSnake 9/10. This gate is diagnostic, not a tuning result.
- A full untuned-Qwen run has not been completed. Therefore the project does not yet
  claim that QLoRA improved the Qwen base on full HumanEval+.
- The project direction, model pair, and benchmark were selected during exploration.
  The result is exploratory rather than preregistered confirmatory evidence.
- Both reported checkpoints are 4-bit community conversions. Quantization,
  conversion, tokenizer, chat-template, and server differences may affect scores.
- HumanEval+ measures isolated Python functions, not repository-level engineering,
  French-language capability, safety, or general reasoning.

Retraining after inspecting these results would make HumanEval+ development data for
this project. Any subsequent tuning claim should be confirmed on a fresh benchmark or
private holdout.

## Reproduce

```bash
uv pip install --python .venv/bin/python -e '.[local,dev]'

HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download \
  mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --local-dir models/qwen3-4b-instruct-2507-4bit
HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download \
  mlx-community/Ministral-3-3B-Instruct-2512-4bit \
  --revision a962dcb09eee4169c890e544c9eb938f1113fdee \
  --local-dir models/ministral-3-3b-instruct-2512-4bit

.venv/bin/python -m data.prepare_mbpp_train
.venv/bin/python -m finetune.run_finetune
.venv/bin/python -m benchmark.run_evalplus --model mistral
.venv/bin/python -m benchmark.run_evalplus --model tuned
```

The evaluation executes model-generated Python. Run it only in an appropriately
isolated environment. On platforms where EvalPlus's address-space limit is
incompatible, this harness disables that memory limit while retaining EvalPlus's
other guards and timeouts.

For the missing causal ablation, run:

```bash
.venv/bin/python -m benchmark.run_evalplus --model qwen
```

## Release status

- Local SheSnake-versus-Ministral evidence: complete
- Full untuned-base ablation: pending
- Public Hugging Face adapter and model card: pending
- Transformers-compatible merged export: pending
- External leaderboard submission or accepted rank: pending

The project builds on Qwen's open foundation with explicit upstream credit—**谢谢
(xièxie)** to the Qwen team. The goal is evidence-efficient specialization: establish
a reproducible narrow result before spending on larger experiments.

See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the release checklist and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for contributions.
