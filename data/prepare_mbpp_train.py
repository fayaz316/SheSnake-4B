"""Prepare checker-verified MBPP train data for the local QLoRA pass."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "code_sft"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def too_similar_to_humaneval(prompt: str, humaneval_prompts: list[str]) -> bool:
    candidate = normalize(prompt)
    return any(
        difflib.SequenceMatcher(None, candidate, normalize(other)).ratio() >= 0.82
        for other in humaneval_prompts
    )


def verify_solution(code: str, setup: str, tests: list[str], timeout: float = 4) -> tuple[bool, str]:
    payload = "\n".join([setup or "", code, *tests])
    try:
        compile(payload, "<mbpp>", "exec")
    except SyntaxError as exc:
        return False, f"syntax: {exc}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        try:
            result = subprocess.run(
                [sys.executable, "-I", handle.name], capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    detail = (result.stderr or result.stdout).strip()[-500:]
    return result.returncode == 0, detail


def as_chat(row: dict) -> dict:
    tests = "\n".join(row.get("test_list") or [])
    prompt = (
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:\n\n"
        f"{row['prompt'].strip()}\n\nThe function must satisfy:\n{tests}"
    )
    return {
        "messages": [
            {"role": "system", "content": "You are a precise Python coding assistant. Return one correct implementation."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```python\n{row['code'].strip()}\n```"},
        ],
        "source": "google-research-datasets/mbpp:sanitized/train",
        "source_task_id": row.get("task_id"),
        "verified": True,
    }


def prepare(output_dir: Path = OUTPUT_DIR, max_examples: int = 320) -> dict[str, int]:
    from datasets import load_dataset
    from evalplus.data import get_human_eval_plus

    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    humaneval_prompts = [item["prompt"] for item in get_human_eval_plus().values()]
    accepted: list[dict] = []
    rejected = 0
    for row in dataset:
        if too_similar_to_humaneval(row["prompt"], humaneval_prompts):
            rejected += 1
            continue
        passed, _ = verify_solution(
            row["code"], row.get("test_setup_code") or "", row.get("test_list") or []
        )
        if not passed:
            rejected += 1
            continue
        accepted.append(as_chat(dict(row)))
        if len(accepted) >= max_examples:
            break
    if len(accepted) < 50:
        raise RuntimeError(f"Only {len(accepted)} verified examples; refusing to train")

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_size = max(16, len(accepted) // 10)
    splits = {"train": accepted[:-validation_size], "valid": accepted[-validation_size:]}
    for split, rows in splits.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": "google-research-datasets/mbpp", "config": "sanitized",
        "source_split": "train", "primary_benchmark": "HumanEval+",
        "accepted": len(accepted), "rejected": rejected,
        "train": len(splits["train"]), "valid": len(splits["valid"]),
        "verification": "compile + supplied assertions in python -I subprocess",
        "prompt_overlap_screen": (
            "no normalized literal-similarity match >= 0.82 against HumanEval+ prompts"
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: manifest[key] for key in ("accepted", "rejected", "train", "valid")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-examples", type=int, default=320)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(max_examples=args.max_examples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
