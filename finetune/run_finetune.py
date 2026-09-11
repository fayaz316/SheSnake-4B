"""Run exactly one conservative local MLX QLoRA configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FinetuneConfig:
    model: Path = ROOT / "models" / "qwen3-4b-instruct-2507-4bit"
    data: Path = ROOT / "data" / "code_sft"
    adapter_path: Path = ROOT / "finetune" / "adapters" / "shesnake-4b"
    iterations: int = 200
    num_layers: int = 8
    batch_size: int = 1
    grad_accumulation_steps: int = 4
    max_seq_length: int = 2048
    learning_rate: float = 1e-5


def build_command(config: FinetuneConfig) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm.lora", "--model", str(config.model),
        "--train", "--data", str(config.data), "--adapter-path", str(config.adapter_path),
        "--fine-tune-type", "lora", "--iters", str(config.iterations),
        "--num-layers", str(config.num_layers), "--batch-size", str(config.batch_size),
        "--grad-accumulation-steps", str(config.grad_accumulation_steps),
        "--max-seq-length", str(config.max_seq_length),
        "--learning-rate", str(config.learning_rate),
        "--steps-per-report", "5", "--steps-per-eval", "25", "--save-every", "50",
        "--mask-prompt", "--grad-checkpoint",
    ]


def validate(config: FinetuneConfig) -> None:
    if not (config.model / "config.json").exists():
        raise FileNotFoundError(f"Base model is not downloaded: {config.model}")
    for split in ("train", "valid"):
        path = config.data / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing verified data: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows or any(not row.get("verified") for row in rows):
            raise ValueError(f"{path} contains empty or unverified data")


def run(config: FinetuneConfig) -> Path:
    validate(config)
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    (config.adapter_path / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8"
    )
    subprocess.run(build_command(config), check=True)
    adapter = config.adapter_path / "adapters.safetensors"
    if not adapter.exists():
        raise RuntimeError(f"Training exited without producing {adapter}")
    return adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = FinetuneConfig(iterations=args.iters)
    if args.dry_run:
        print(" ".join(build_command(config)))
        return 0
    print(run(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
