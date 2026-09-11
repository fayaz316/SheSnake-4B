# SheSnake-4B: an exploratory study in compute-efficient specialization

**Bonjour.**

## Abstract

SheSnake-4B is a parameter-efficient adaptation of Qwen3-4B-Instruct-2507 for
compact Python function generation. A 3.67M-parameter QLoRA adapter was trained on 104
checker-verified examples from the sanitized MBPP training split and evaluated with
EvalPlus 0.3.1. Under a matched local protocol, the adapted 4-bit MLX checkpoint
scored 79.3% on HumanEval+ versus 71.3% for the specified 4-bit MLX conversion of
Ministral-3-3B-Instruct-2512. The result supports a narrow model-comparison claim; it
does not by itself show that adaptation improved the Qwen base because a full
untuned-base ablation has not yet been completed. The experiment is reported as
exploratory because the model pair and benchmark were selected during project
iteration rather than preregistered.

## 1. Research question and contribution

The study asks whether a small, verified supervised set and a reproducible local
workflow can produce useful narrow-domain evidence without paid inference or rented
accelerators. Its contributions are:

1. a compact QLoRA adapter over an openly licensed upstream model;
2. a data-admission pipeline that compiles and executes each supplied reference
   solution before accepting it;
3. a matched generation and executable-evaluation path for two compact checkpoints;
4. public documentation of protocol, provenance, limitations, and release status.

This is not foundation-model pretraining. Qwen provides the underlying model; the
project contribution is the adapter, data pipeline, evaluation harness, and evidence.

## 2. Model identity

- Release: **SheSnake-4B**
- Upstream base: `Qwen/Qwen3-4B-Instruct-2507`
- Training checkpoint: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- Comparison checkpoint: `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- Adaptation: MLX QLoRA
- Specialization: English-prompted Python function generation
- Provenance: training executed in France

Training was performed in France. This is provenance, not evidence of French-language
specialization. Both upstream models are Apache-2.0 licensed. The reported scores
apply to the named community conversions, not automatically to the upstream
full-precision checkpoints.

### Recorded conversion revisions

- Qwen MLX: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
- Ministral MLX: `a962dcb09eee4169c890e544c9eb938f1113fdee`

## 3. Training data

The source is `google-research-datasets/mbpp`, configuration `sanitized`, split
`train`, licensed CC BY 4.0. The source split contained 120 examples in the local
dataset revision. All 120 compiled and passed their supplied assertions; 104 were
used for optimization and the final 16 were retained as validation data.

Each MBPP prompt was compared with all HumanEval+ prompts after lowercasing and
whitespace normalization. No pair reached the exclusion threshold of 0.82 under
`difflib.SequenceMatcher`, so zero examples were removed. This procedure detects
high literal similarity only. It is not semantic decontamination and cannot establish
that related code or tasks were absent from upstream pretraining.

Reference solutions were checked in a Python isolated-mode subprocess with a
four-second timeout. This protects the preparation process from common failures but
is not an operating-system security sandbox.

## 4. Adaptation procedure

The base checkpoint remained frozen. The single recorded run used:

| Setting | Value |
|---|---:|
| Trainable parameters | approximately 3.67M / 4.02B (0.091%) |
| Optimizer | Adam |
| Steps | 200 |
| Effective batch | 4 (batch 1; accumulation 4) |
| Adapted layers | 8 |
| LoRA rank / scale / dropout | 8 / 20 / 0 |
| Maximum sequence length | 2,048 |
| Learning rate | `1e-5` |
| Seed | 0 |
| Prompt masking | enabled |
| Gradient checkpointing | enabled |

The final recorded validation loss was 0.691. Loss is an optimization diagnostic and
must not be read as accuracy, pass@1, or a leaderboard score. No hyperparameter sweep
or checkpoint selection on the full HumanEval+ result was performed.

## 5. Evaluation protocol

The primary benchmark was EvalPlus 0.3.1 HumanEval+:

- all 164 tasks;
- one completion per task;
- greedy decoding at temperature 0;
- a 768-new-token cap;
- one EvalPlus prompt builder and output sanitizer;
- executable HumanEval and HumanEval+ tests;
- complete outputs and evaluation records for both reported checkpoints.

Each model retained its own tokenizer, chat template, and server implementation.
SheSnake used `mlx-lm==0.31.3`; Ministral used `mlx-vlm==0.7.0`. This is a practical
matched application-level comparison, not a controlled full-precision architecture
ablation. The full environment also used `datasets==5.0.1` and
`transformers==5.17.0`.

## 6. Results

| Local checkpoint | HumanEval pass@1 | HumanEval+ pass@1 |
|---|---:|---:|
| SheSnake-4B | **140/164 (85.4%)** | **130/164 (79.3%)** |
| Ministral-3-3B-Instruct-2512-4bit | 124/164 (75.6%) | 117/164 (71.3%) |

HumanEval+ paired outcomes were:

| Outcome | Tasks |
|---|---:|
| Both pass | 109 |
| SheSnake only | 21 |
| Ministral only | 8 |
| Neither passes | 26 |

The observed HumanEval+ difference is 13 tasks, or 8.0 percentage points. A
descriptive paired bootstrap with 100,000 resamples and seed 0 produced a 95% interval
of approximately +1.8 to +14.0 points; an exact two-sided McNemar test on the 29
discordant tasks gives `p=0.024`. Because the target and benchmark were selected
during exploration, these statistics are descriptive rather than a confirmatory
hypothesis test.

## 7. Integrity assessment

The result is real under the recorded protocol:

- HumanEval and HumanEval+ solutions were not supervised training targets.
- Both reported models received all 164 tasks through the same generation path.
- No reported advantage comes from missing outputs or different task counts.
- Every score was produced by executable tests rather than subjective grading.
- Raw generations and evaluator outputs are retained locally; privacy-safe aggregate
  artifacts are included in `results/`.

The following caveats prevent broader claims:

1. **No full untuned-base ablation.** Untuned Qwen and SheSnake each scored 9/10 on
   the diagnostic gate. A full Qwen result is required to attribute any full-benchmark
   change to the adapter.
2. **Exploratory selection.** The project moved across model sizes and candidate
   comparisons before locking this compact track. The SheSnake-versus-Ministral
   result should not be presented as preregistered.
3. **Public-benchmark contamination.** Upstream pretraining exposure to HumanEval,
   HumanEval+, MBPP, or close derivatives cannot be established from public model
   cards.
4. **Conversion confounds.** Both models are community 4-bit conversions and use
   different model-specific serving packages.
5. **Limited construct validity.** HumanEval+ tests isolated Python functions, not
   repository maintenance, software design, security, general reasoning, or French
   language ability.
6. **Single run and seed.** Training variance and robustness across seeds were not
   measured.

## 8. Safe interpretation

The supported conclusion is that the released SheSnake configuration outperformed
the specified Ministral conversion on one complete, matched local HumanEval+ run. The
experiment demonstrates that a compact, low-cash-cost workflow can produce an
auditable result quickly.

It does not establish that QLoRA caused the advantage, that SheSnake is generally
better than Ministral, that it is a national or category leader, or that an external
leaderboard has accepted the model.

## 9. Next experiments

1. Run the untouched Qwen checkpoint on all 164 tasks to measure the adapter delta.
2. Freeze the current adapter and confirm on a fresh benchmark such as a versioned
   LiveCodeBench slice or a private holdout; do not retrain against HumanEval+.
3. Repeat training across multiple seeds and report mean, variance, and failures.
4. Evaluate repository-level coding and French-language capability as separate
   studies with task-appropriate data.
5. Export a Transformers-compatible Safetensors model before submitting to a board
   that requires Hugging Face AutoClass loading.

## 10. Acknowledgements

The project thanks the Qwen team and open-model ecosystem—**谢谢 (xièxie)**—for the
upstream foundation. Qwen and its authors remain credited as upstream contributors.

## Sources

1. Qwen Team, [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
2. Mistral AI, [Ministral-3-3B-Instruct-2512 model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).
3. MLX team, [MLX-LM fine-tuning guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
4. EvalPlus authors, [EvalPlus](https://github.com/evalplus/evalplus).
5. Google Research, [MBPP dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp).
6. Hugging Face, [Open LLM Leaderboard submission requirements](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/submitting).
