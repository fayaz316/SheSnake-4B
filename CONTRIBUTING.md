# Contributing

Contributions are welcome when they improve reproducibility, evaluation quality,
documentation, or compact-model specialization.

## Workflow

1. Fork the GitHub repository.
2. Create a focused branch.
3. Make the change with tests or an explicit reproduction note.
4. Run `pytest -q`.
5. Open a pull request describing the motivation, changed files, and evidence.

The GitHub repository is the canonical home for source code and documentation. The
Hugging Face repository is the release home for model artifacts and their model card;
artifact changes should link to the corresponding source change.

## Evaluation standards

- Keep benchmark prompts, decoding settings, and execution rules matched.
- Do not train on held-out evaluation solutions.
- Report negative results and failed experiments when they affect interpretation.
- Preserve upstream licenses and attribution.
- Do not include credentials, private paths, personal identifiers, or local caches.
