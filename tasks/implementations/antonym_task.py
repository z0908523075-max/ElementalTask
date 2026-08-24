"""Antonym elemental task — map a word to its antonym.

This is a Phase-I elemental ICL task (see README "Synonyms/antonyms").
The model is shown demonstration of the form `Input: X\nOutput: Y` where
`Y` is the antonym of `X`, then queried with a new `X`.

Follows the same in-memory / registry pattern as `token_reversal.py` and
`copying_task.py`.
"""

from typing import Dict, List, Optional

from tasks.base_task import BaseTask, TaskConfig


class AntonymTask(BaseTask):
    """Task for mapping a word to its antonym."""

    TASK_NAME = "antonym"  # automatic discovery key

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions using exact-match accuracy (case-insensitive, trim leading and trailing whitespace)."""
        ground_truth = self.get_ground_truth(split)

        if len(predictions) != len(ground_truth):
            return {
                "accuracy": 0.0,
                "error": (
                    f"Prediction count ({len(predictions)}) does not match "
                    f"ground truth count ({len(ground_truth)})"
                ),
            }

        processed_predictions = [self.preprocess_prediction(p) for p in predictions]
        correct = sum(
            1 for pred, gt in zip(processed_predictions, ground_truth)
            if pred.lower().strip() == str(gt).lower().strip()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth),
        }


# default antonym pairs — kept small, unambiguous, single token where possible.
_DEFAULT_ANTONYM_PAIRS: List[Dict[str, str]] = [
    {"input": "hot", "output": "cold"},
    {"input": "big", "output": "small"},
    {"input": "fast", "output": "slow"},
    {"input": "happy", "output": "sad"},
    {"input": "light", "output": "dark"},
    {"input": "up", "output": "down"},
    {"input": "open", "output": "closed"},
    {"input": "rich", "output": "poor"},
    {"input": "strong", "output": "weak"},
    {"input": "young", "output": "old"},
    {"input": "new", "output": "old"},
    {"input": "easy", "output": "hard"},
    {"input": "clean", "output": "dirty"},
    {"input": "full", "output": "empty"},
    {"input": "high", "output": "low"},
    {"input": "wet", "output": "dry"},
    {"input": "soft", "output": "hard"},
    {"input": "near", "output": "far"},
    {"input": "true", "output": "false"},
    {"input": "day", "output": "night"},
    {"input": "sharp", "output": "dull"},
    {"input": "tight", "output": "loose"},
    {"input": "smooth", "output": "rough"},
    {"input": "brave", "output": "cowardly"},
    {"input": "wide", "output": "narrow"},
]


def create_antonym_task(
    examples: Optional[List[Dict[str, str]]] = None,
    name: str = "antonym",
) -> AntonymTask:
    """Create a ``AntonymTask`` instance.

    Args: 
        examples: Optional list of ``{"input": word, "output": antonym}`` dicts.
                  If None, uses a built-in default set.
        name: task name to register/report.

    Returns: 
        An initialized ``AntonymTask``.
    """
    config = TaskConfig(
        name=name,
        description="Antonym elemental task: map a word to its antonym",
        data_format="memory",
        in_memory_data=examples if examples is not None else _DEFAULT_ANTONYM_PAIRS,
        input_column="input",
        output_column="output",
        prompt_template="Input: {input}\nOutput:",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "lexical_semantics"},
    )
    return AntonymTask(config)
