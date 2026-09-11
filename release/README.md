---
language:
- en
library_name: mlx
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
- google-research-datasets/mbpp
tags:
- qlora
- coding
- python
- fine-tuned
license: apache-2.0
---

# SheSnake-4B 🇫🇷

**Bonjour.**

SheSnake-4B is an exploratory 🇫🇷 MLX QLoRA adapter for
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
specialized on compact Python function-generation examples. It was trained locally in
France. This is training provenance, not a claim of French-language specialization.

This is an adapter derivative, not a foundation model trained from scratch.

## Results

| Local checkpoint | HumanEval | HumanEval+ |
|---|---:|---:|
| SheSnake-4B | **140/164 (85.4%)** | **130/164 (79.3%)** |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | 124/164 (75.6%) | 117/164 (71.3%) |

Evaluation used `evalplus==0.3.1`, one greedy completion per task, temperature 0, a
768-new-token cap, and the same EvalPlus prompt, sanitization, and executable scoring
path. The observed HumanEval+ difference was +13 tasks, or +8.0 percentage points.

These are matched local measurements for the exact 4-bit community checkpoints—not
an official leaderboard placement and not a claim about all Mistral or Qwen variants.
Method, code, and aggregate evidence are available at
[`abrarf316/SheSnake-4B`](https://github.com/abrarf316/SheSnake-4B).

## Training

- Base checkpoint: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- Base revision: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
- Method: QLoRA; base weights frozen
- Trainable parameters: approximately 3.67M / 4.02B (0.091%)
- Data: 104 verified MBPP `sanitized/train` examples
- Validation: 16 examples from the same source split
- Steps: 200; effective batch 4; learning rate `1e-5`; 8 adapted layers
- LoRA: rank 8, scale 20, dropout 0; seed 0

Each admitted solution compiled and passed the assertions supplied with its MBPP
record. The source data is CC BY 4.0; see the repository's license notice.

## Evaluation integrity

- No HumanEval or HumanEval+ solutions were used as supervised targets.
- All 164 tasks produced an output and evaluation record for each reported model.
- MBPP prompts were screened for normalized literal similarity of 0.82 or greater
  against HumanEval+ prompts; no examples crossed the threshold. This is not proof of
  semantic decontamination.
- Public benchmark exposure during upstream pretraining cannot be ruled out.
- The first 10 HumanEval+ tasks were used as a systems gate: Ministral 8/10, untuned
  Qwen 9/10, and SheSnake 9/10.
- A full untuned-Qwen run has not been completed. No claim is made that this adapter
  improves the Qwen base across full HumanEval+.
- The model pair and benchmark were selected during exploration, so this is an
  exploratory result rather than a preregistered confirmatory experiment.
- The two checkpoints use their own tokenizers, chat templates, and MLX serving
  packages; conversion and quantization effects remain possible confounders.

Retraining after viewing this result would turn HumanEval+ into development data for
this project. Future tuning should be confirmed on a fresh benchmark or private
holdout.

## Intended use

This adapter is intended for research and experimentation with compact Python code
generation. Human review and testing remain necessary. It is not validated for
security-critical, safety-critical, or production code generation.

## Limitations

HumanEval+ measures isolated functions. It does not establish repository-level
software-engineering ability, French-language capability, general reasoning quality,
or broad superiority over another model family. Training variance across seeds has
not been measured.

## Loading

Use the adapter with the named MLX base checkpoint. Exact commands and dependency
versions are maintained in the GitHub repository. An MLX adapter is not automatically
compatible with leaderboards that require a standalone Transformers AutoClass model;
a separate merged Safetensors export is planned.

## License and attribution

The adapter code and original project materials are provided under Apache-2.0. The
Qwen base remains subject to its upstream Apache-2.0 license. The included MBPP-derived
training records are attributed under CC BY 4.0.

**谢谢 (xièxie)** to the Qwen team and upstream contributors for making the foundation
available. Feedback and reproducible pull requests are welcome.
