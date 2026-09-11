"""Run matched EvalPlus generation and executable evaluation through local MLX servers.

All models share one compatibility endpoint contract, prompt builder, output sanitizer,
token cap, and executable evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "eval" / "evalplus_results"


@dataclass(frozen=True)
class LocalModel:
    name: str
    path: Path
    server_module: str
    adapter_path: Path | None = None


MODELS = {
    "qwen": LocalModel(
        name="qwen3-4b-instruct-2507",
        path=ROOT / "models" / "qwen3-4b-instruct-2507-4bit",
        server_module="mlx_lm.server",
    ),
    "mistral": LocalModel(
        name="ministral-3-3b-instruct-2512",
        path=ROOT / "models" / "ministral-3-3b-instruct-2512-4bit",
        server_module="mlx_vlm.server",
    ),
    "tuned": LocalModel(
        name="shesnake-4b",
        path=ROOT / "models" / "qwen3-4b-instruct-2507-4bit",
        server_module="mlx_lm.server",
        adapter_path=ROOT / "finetune" / "adapters" / "shesnake-4b",
    ),
}


def server_command(model: LocalModel, port: int) -> list[str]:
    cmd = [
        sys.executable, "-m", model.server_module,
        "--model", str(model.path), "--host", "127.0.0.1", "--port", str(port),
    ]
    if model.adapter_path is not None:
        cmd.extend(["--adapter-path", str(model.adapter_path)])
    return cmd


def _wait_for_server(port: int, process: subprocess.Popen, timeout: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MLX server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"MLX server did not become ready within {timeout}s")


def _assert_ready(model: LocalModel) -> None:
    if not (model.path / "config.json").exists():
        raise FileNotFoundError(f"Model is not downloaded: {model.path}")
    if model.adapter_path is not None and not (model.adapter_path / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Adapter is not trained: {model.adapter_path}")


def _write_subset(dataset: str, limit: int, root: Path) -> Path:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl

    problems = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    subset_path = root / f"{dataset}-first-{limit}.jsonl"
    write_jsonl(str(subset_path), list(problems.values())[:limit], drop_builtin=False)
    return subset_path


def run(model_key: str, dataset: str, limit: int | None, port: int) -> Path:
    from evalplus.codegen import run_codegen

    model = MODELS[model_key]
    _assert_ready(model)
    run_kind = f"gate{limit}" if limit else "full"
    root = RESULTS_ROOT / run_kind / model_key
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "server.log"
    subset_path = _write_subset(dataset, limit, root) if limit else None

    with log_path.open("a", encoding="utf-8") as log:
        server = subprocess.Popen(
            server_command(model, port), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(port, server)
            old_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "local-no-key"
            try:
                samples = Path(run_codegen(
                    model=str(model.path), dataset=dataset, root=str(root),
                    n_samples=1, temperature=0.0, greedy=True,
                    id_range=(0, limit) if limit else None, backend="openai",
                    base_url=f"http://127.0.0.1:{port}/v1", jsonl_fmt=True,
                ))
            finally:
                if old_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_key
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)

    eval_env = os.environ.copy()
    # EvalPlus 0.3.1 attempts to raise Darwin's hard RLIMIT_AS to 4 GiB. Passing
    # -1 disables only that incompatible memory limit; its destructive-operation
    # reliability guard and per-test timeouts remain active.
    eval_env["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
    if subset_path is not None:
        override = "HUMANEVAL_OVERRIDE_PATH" if dataset == "humaneval" else "MBPP_OVERRIDE_PATH"
        eval_env[override] = str(subset_path)
    subprocess.run([
        sys.executable, "-m", "evalplus.evaluate", dataset,
        "--samples", str(samples), "--parallel", "4",
    ], check=True, env=eval_env)
    metadata = {
        "model_key": model_key, "model_name": model.name,
        "model_path": str(model.path),
        "adapter_path": str(model.adapter_path) if model.adapter_path else None,
        "dataset": dataset, "limit": limit,
        "decoding": {
            "temperature": 0.0, "n_samples": 1, "greedy": True,
            "max_new_tokens": 768,
        },
        "evalplus_version": __import__("evalplus").__version__,
        "samples": str(samples),
    }
    (root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--limit", type=int, help="Run task IDs [0, limit) as a gate")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.dry_run:
        print(" ".join(server_command(MODELS[args.model], args.port)))
        return 0
    print(run(args.model, args.dataset, args.limit, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
