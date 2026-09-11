# Dataset attribution and transformations

The files in this directory are derived from the sanitized training split of
**Mostly Basic Python Problems (MBPP)**, curated by Google Research and distributed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source: [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp)

Citation:

> Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski,
> David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. “Program
> Synthesis with Large Language Models.” arXiv:2108.07732, 2021.

## Changes made in this repository

- selected records from the `sanitized/train` partition;
- executed each reference solution against its supplied assertions;
- screened prompts for high normalized literal similarity to HumanEval+ prompts;
- wrapped prompts, tests, and reference code in a chat-style supervised format;
- added source identifiers and verification metadata;
- assigned 104 records to training and 16 to validation.

The original task descriptions, reference code, and supplied assertions remain
attributable to the MBPP dataset. Google Research and the MBPP authors do not endorse
SheSnake-4B or this adaptation.
