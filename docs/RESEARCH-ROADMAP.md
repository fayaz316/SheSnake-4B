# Research roadmap

SheSnake-4B is one controlled case study: a compact model, a narrow Python-coding
target, verified supervised data, and executable evaluation. The general method is
reusable, but every new benchmark requires a fresh protocol rather than automatic
transfer of the headline score.

## A repeatable specialization loop

1. **Define the capability.** State the exact behavior to improve and the intended
   users. “Coding” is too broad; isolated Python functions, test repair, repository
   navigation, and code review are different capabilities.
2. **Select a matched benchmark.** Use an official or well-documented benchmark with
   a public protocol, executable or objectively graded outcomes, and a held-out test
   path.
3. **Build verified training data.** Use only permitted training partitions, validate
   every example, and record provenance. Synthetic data can be useful, but its source,
   generator, and filtering rules must be disclosed.
4. **Run the unchanged baseline.** Establish the starting score before adaptation.
5. **Adapt one controlled configuration.** Keep the first experiment simple enough to
   reproduce and avoid silently selecting a favorable checkpoint.
6. **Evaluate with the same rules.** Match prompts, sampling, token limits, tools,
   test execution, and scoring across every comparison model.
7. **Publish the full delta.** Report baseline, adapted model, data, failures,
   limitations, and resource use—not only the best number.

## Benchmark-specific extensions

| Target | Useful evidence | Main caution |
|---|---|---|
| Function-level Python coding | HumanEval+, MBPP+ | Public-task contamination and limited scope |
| Fresh competitive programming | LiveCodeBench | Time windows, version pinning, and higher run cost |
| Repository-level engineering | SWE-bench-style tasks or controlled local repositories | Tool use, patch validity, and environment differences |
| Mathematics | GSM8K, MATH, or newer held-out sets | Exact answer parsing, reasoning traces, and contamination |
| French capability | French instruction, coding, and knowledge suites | Translation quality is not the same as native reasoning |
| General instruction following | IFEval-style tests and human-reviewed subsets | Automated checks do not capture every quality dimension |

The benchmark should be selected before tuning. If several benchmarks are explored,
their results should be reported as a portfolio rather than using only the most
favorable task.

## What changes at larger parameter scales

The learning objective remains the same: minimize task loss on permitted examples and
measure executable or objective performance on held-out data. The engineering and
experimental discipline change substantially:

- **Compact models:** local QLoRA, small verified datasets, rapid iteration, and
  careful memory limits are practical.
- **Medium models:** distributed or rented GPUs, larger activation memory, stronger
  checkpoint management, and multiple seeds become more important.
- **Very large models:** data curation, optimizer state, parallelism, fault recovery,
  inference cost, and evaluation throughput dominate. Adapter tuning may remain the
  right method, but it is no longer automatically cheap.

Scaling should follow evidence. First establish that the data and task definition
produce a real gain on a compact baseline. Then increase model size, data volume, or
training budget one factor at a time so the source of any improvement remains
identifiable.

## A responsible improvement claim

A strong claim has the form:

> Under protocol P, on benchmark B and revision R, adaptation method A changed the
> matched baseline from X to Y, with data D, compute budget C, and limitations L.

This structure makes the result useful even when the adapted model does not win. It
lets others reproduce the experiment, challenge the assumptions, and build on the
parts that worked.

## Sources

- [EvalPlus](https://github.com/evalplus/evalplus)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [MLX-LM fine-tuning guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
