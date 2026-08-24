"""Antonym elemental 任務 — map a 詞 to its antonym.

This is a Phase-I elemental ICL 任務 (see README "Synonyms/antonyms").
The 模型 is shown 示範 of the form `Input: X\nOutput: Y` where
`Y` is the antonym of `X`, then queried with a 新的 `X`.

Follows the 相同 記憶體內 / registry pattern as `token_reversal.py` and
`copying_task.py`.
"""

from typing import Dict, List, Optional

from tasks.base_task import BaseTask, TaskConfig


class AntonymTask(BaseTask):
    """用於mapping a 詞 to its antonym."""

    TASK_NAME = "antonym"  # 自動發現鍵

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """以完全匹配準確率評估預測 (不區分大小寫, 去除首尾空白)."""
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


# 預設 antonym pairs — kept small, unambiguous, 單 token where possible.
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
    """建立一個``AntonymTask`` 實例.

    參數：
        範例: 可選list of ``{"input": word, "output": antonym}`` dicts.
                  If None, uses a built-in 預設set.
        名稱: 任務 名稱 to 註冊/report.

    回傳：
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
