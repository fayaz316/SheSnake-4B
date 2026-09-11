# License and attribution notice

Original source code and documentation in this repository are released under the
Apache License 2.0; see `LICENSE`.

The adapter is a derivative of **Qwen/Qwen3-4B-Instruct-2507**, released under the
**Apache License 2.0**. Training and evaluation were performed in France.

- Qwen and Alibaba's Qwen Team remain clearly credited as the upstream authors.
- A release must include the upstream license and a notice describing our adapter,
  training data, and modifications.
- “Fine-tuned in France” is accurate; “French-language model” and “trained from
  scratch in France” are not.

The comparison checkpoint, **mistralai/Ministral-3-3B-Instruct-2512**, is also
Apache-2.0 and is used only for reproducible evaluation.

The committed training records are derived from the
[`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp)
sanitized training split, published under **CC BY 4.0**. They retain source identifiers
and provenance fields for attribution. The MBPP authors and Google Research do not
endorse this derivative model.

EvalPlus is used under its Apache-2.0 license. HumanEval material distributed through
that evaluation path remains subject to its source terms. Generated benchmark outputs
are published as research evidence and do not change the licenses of benchmark text.
