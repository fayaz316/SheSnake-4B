from benchmark.export_public_results import exact_mcnemar_p, status_map
from benchmark.run_evalplus import MODELS, server_command
from data.prepare_mbpp_train import as_chat, normalize, verify_solution
from finetune.run_finetune import FinetuneConfig, build_command


def test_local_model_servers_are_explicit_and_matched():
    qwen = server_command(MODELS["qwen"], 9000)
    mistral = server_command(MODELS["mistral"], 9000)
    assert "mlx_lm.server" in qwen
    assert "mlx_vlm.server" in mistral
    assert qwen[-1] == mistral[-1] == "9000"


def test_tuned_model_loads_the_only_adapter():
    command = server_command(MODELS["tuned"], 9000)
    assert "--adapter-path" in command
    assert command.count("--adapter-path") == 1
    assert MODELS["tuned"].name == "shesnake-4b"


def test_training_verifier_executes_supplied_assertions():
    passed, _ = verify_solution("def add(a, b): return a + b", "", ["assert add(2, 3) == 5"])
    failed, _ = verify_solution("def add(a, b): return a - b", "", ["assert add(2, 3) == 5"])
    assert passed
    assert not failed


def test_training_row_is_chat_formatted_and_marked_verified():
    row = as_chat({
        "task_id": 1,
        "prompt": "Add two integers.",
        "code": "def add(a, b): return a + b",
        "test_list": ["assert add(1, 2) == 3"],
    })
    assert row["verified"] is True
    assert row["messages"][-1]["role"] == "assistant"
    assert normalize(" A  B\n") == "a b"


def test_finetune_command_is_single_conservative_qlora_pass():
    command = build_command(FinetuneConfig())
    assert command.count("--train") == 1
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--num-layers") + 1] == "8"
    assert "--grad-checkpoint" in command
    assert "--mask-prompt" in command


def test_humaneval_plus_requires_base_and_added_tests_to_pass():
    evaluation = {
        "eval": {
            "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
            "HumanEval/1": [{"base_status": "fail", "plus_status": "pass"}],
        }
    }
    base, plus = status_map(evaluation)
    assert base == {"HumanEval/0": True, "HumanEval/1": False}
    assert plus == {"HumanEval/0": True, "HumanEval/1": False}


def test_exact_mcnemar_statistic_matches_release_scorecard():
    assert exact_mcnemar_p(21, 8) == 0.0241195447742939
