"""Unit tests for the AntonymTask elemental task."""

import pytest

from tasks.implementations.antonym_task import (
    AntonymTask,
    create_antonym_task,
    _DEFAULT_ANTONYM_PAIRS,
)


def test_create_antonym_task_defaults():
    task = create_antonym_task()
    assert isinstance(task, AntonymTask)
    assert task.TASK_NAME == "antonym"
    assert task.config.name == "antonym"
    rows = task.get_split("test")
    assert len(rows) == len(_DEFAULT_ANTONYM_PAIRS)
    assert rows[0]["input"] == "hot"
    assert rows[0]["output"] == "cold"


def test_create_antonym_task_custom_examples():
    custom = [
        {"input": "good", "output": "bad"},
        {"input": "yes", "output": "no"},
    ]
    task = create_antonym_task(examples=custom)
    rows = task.get_split("test")
    assert len(rows) == 2
    assert rows[1]["input"] == "yes"


def test_build_prompt_uses_template():
    task = create_antonym_task()
    instance = {"input": "hot", "output": "cold"}
    prompt = task.build_prompt(instance, num_shots=0)
    # prompt_template should place the input after "Input:"
    assert "Input: hot" in prompt
    assert prompt.rstrip().endswith("Output:")


def test_get_icl_examples_shape():
    task = create_antonym_task()
    examples = task.get_icl_examples(num_examples=5, seed=0)
    assert len(examples) == 5
    for ex in examples:
        assert "input" in ex and "output" in ex
        assert isinstance(ex["input"], str) and isinstance(ex["output"], str)


def test_evaluate_perfect_predictions():
    task = create_antonym_task()
    gt = task.get_ground_truth("test")
    result = task.evaluate(gt)
    assert result["accuracy"] == 1.0
    assert result["correct"] == result["total"] == len(gt)


def test_evaluate_case_insensitive_and_trimmed():
    task = create_antonym_task()
    gt = task.get_ground_truth("test")
    noisy = [f"  {g.upper()}  " for g in gt]
    result = task.evaluate(noisy)
    assert result["accuracy"] == 1.0


def test_evaluate_wrong_predictions():
    task = create_antonym_task()
    gt = task.get_ground_truth("test")
    wrong = ["definitely_wrong"] * len(gt)
    result = task.evaluate(wrong)
    assert result["accuracy"] == 0.0
    assert result["correct"] == 0


def test_evaluate_length_mismatch_returns_error():
    task = create_antonym_task()
    result = task.evaluate(["cold"])  # far fewer than ground truth
    assert result["accuracy"] == 0.0
    assert "error" in result


def test_registry_discovers_antonym_task():
    """The registry auto-discovers task via TASK_NAME; ensure ours is found."""
    from tasks.registry import get_task_class, discover_tasks

    discover_tasks()
    cls = get_task_class("antonym")
    assert cls is AntonymTask
