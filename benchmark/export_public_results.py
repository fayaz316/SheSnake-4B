"""Export privacy-safe EvalPlus evidence and a machine-readable scorecard."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "evalplus_results" / "full"
DESTINATION = ROOT / "results"

MODELS = {
    "ministral": {
        "source": "mistral",
        "display_name": "Ministral-3-3B-Instruct-2512-4bit",
        "checkpoint": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        "revision": "a962dcb09eee4169c890e544c9eb938f1113fdee",
    },
    "shesnake-4b": {
        "source": "tuned",
        "display_name": "SheSnake-4B",
        "checkpoint": "mlx-community/Qwen3-4B-Instruct-2507-4bit + QLoRA adapter",
        "revision": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_map(evaluation: dict) -> tuple[dict[str, bool], dict[str, bool]]:
    base: dict[str, bool] = {}
    plus: dict[str, bool] = {}
    for task_id, samples in evaluation["eval"].items():
        if len(samples) != 1:
            raise ValueError(f"Expected one sample for {task_id}, found {len(samples)}")
        sample = samples[0]
        base[task_id] = sample["base_status"] == "pass"
        plus[task_id] = base[task_id] and sample["plus_status"] == "pass"
    return base, plus


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    tail = min(left_only, right_only)
    if discordant == 0:
        return 1.0
    probability = 2 * sum(math.comb(discordant, i) for i in range(tail + 1)) / 2**discordant
    return min(1.0, probability)


def locate(source_key: str) -> tuple[Path, Path, Path]:
    root = SOURCE / source_key / "humaneval"
    raw = next(root.glob("*.raw.jsonl"))
    evaluation = next(root.glob("*_eval_results.json"))
    sanitized = next(path for path in root.glob("*.jsonl") if not path.name.endswith(".raw.jsonl"))
    return raw, sanitized, evaluation


def export() -> Path:
    generations = DESTINATION / "generations"
    evaluations = DESTINATION / "evaluations"
    generations.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)

    scorecard: dict = {
        "release": "SheSnake-4B",
        "status": "exploratory_local_result",
        "benchmark": "EvalPlus 0.3.1 HumanEval+",
        "task_count": 164,
        "decoding": {
            "samples_per_task": 1,
            "greedy": True,
            "temperature": 0.0,
            "max_new_tokens": 768,
        },
        "models": {},
        "integrity": {
            "humaneval_solutions_used_for_training": False,
            "full_untuned_qwen_ablation_complete": False,
            "external_leaderboard_result": False,
        },
    }

    plus_maps: dict[str, dict[str, bool]] = {}
    for public_key, metadata in MODELS.items():
        raw, sanitized, evaluation_path = locate(metadata["source"])
        public_raw = generations / f"{public_key}.raw.jsonl"
        public_sanitized = generations / f"{public_key}.jsonl"
        public_evaluation = evaluations / f"{public_key}.json"
        shutil.copyfile(raw, public_raw)
        shutil.copyfile(sanitized, public_sanitized)
        shutil.copyfile(evaluation_path, public_evaluation)

        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        base, plus = status_map(evaluation)
        if len(base) != 164:
            raise ValueError(f"{public_key} has {len(base)} tasks, expected 164")
        plus_maps[public_key] = plus
        base_passes = sum(base.values())
        plus_passes = sum(plus.values())
        scorecard["models"][public_key] = {
            "display_name": metadata["display_name"],
            "checkpoint": metadata["checkpoint"],
            "checkpoint_revision": metadata["revision"],
            "humaneval": {"passes": base_passes, "total": 164, "pass_at_1": base_passes / 164},
            "humaneval_plus": {"passes": plus_passes, "total": 164, "pass_at_1": plus_passes / 164},
            "artifacts": {
                "raw_generations": {"path": str(public_raw.relative_to(ROOT)), "sha256": sha256(public_raw)},
                "sanitized_generations": {
                    "path": str(public_sanitized.relative_to(ROOT)),
                    "sha256": sha256(public_sanitized),
                },
                "evaluation": {"path": str(public_evaluation.relative_to(ROOT)), "sha256": sha256(public_evaluation)},
            },
        }

    ministral = plus_maps["ministral"]
    shesnake = plus_maps["shesnake-4b"]
    task_ids = sorted(ministral)
    both = sum(ministral[key] and shesnake[key] for key in task_ids)
    shesnake_only = sum(not ministral[key] and shesnake[key] for key in task_ids)
    ministral_only = sum(ministral[key] and not shesnake[key] for key in task_ids)
    neither = sum(not ministral[key] and not shesnake[key] for key in task_ids)
    scorecard["paired_humaneval_plus"] = {
        "both_pass": both,
        "shesnake_only": shesnake_only,
        "ministral_only": ministral_only,
        "neither_pass": neither,
        "delta_tasks": shesnake_only - ministral_only,
        "delta_percentage_points": 100 * (shesnake_only - ministral_only) / 164,
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(shesnake_only, ministral_only),
    }

    destination = DESTINATION / "scorecard.json"
    destination.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(export())
