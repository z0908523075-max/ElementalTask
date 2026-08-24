"""Ignoring 上下文 任務 - Information retrieval with distractors.

Tests the 模型's ability to 擷取 relevant information while ignoring
irrelevant 上下文. Similar to "needle in a haystack" but simpler.

格式：
    [Filler 文字] KEY FACT [More filler] 問題: [Query about KEY FACT]
    答案: [Extracted information]

範例：
    Some 文字 here. X = 5. More 文字 here.
    問題: What is X?
    答案: 5

This tests selective attention and information retrieval capabilities.
"""

import random
import string
from typing import Dict, List, Optional
from tasks.base_task import BaseTask, TaskConfig


class IgnoringContextTask(BaseTask):
    """
    任務 testing the ability to 擷取 key information from irrelevant 上下文.
    
    The 模型 must find and 擷取 a 變數 assignment buried in filler 文字.
    """
    TASK_NAME = "ignoring_context"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False):
        """
        初始化IgnoringContextTask.
        
        參數：
            設定: TaskConfig with 任務 settings
            use_generator: If True, 生成synthetic 範例 with 隨機 變數
        """
        self.use_generator = use_generator
        super().__init__(config)
    
    def _generate_filler_text(self, num_words: int = 5, seed: Optional[int] = None) -> str:
        """生成random filler 文字."""
        if seed is not None:
            random.seed(seed)
        
        # 簡單 詞 for filler
        filler_words = [
            "the", "a", "is", "was", "are", "were", "has", "have", "had",
            "today", "yesterday", "tomorrow", "here", "there", "then", "now",
            "cat", "dog", "tree", "house", "car", "book", "day", "time",
            "good", "bad", "nice", "big", "small", "red", "blue", "green",
            "went", "came", "saw", "said", "made", "took", "gave", "found"
        ]
        
        words = [random.choice(filler_words) for _ in range(num_words)]
        return " ".join(words).capitalize() + "."
    
    def generate_example(
        self,
        variable: Optional[str] = None,
        value: Optional[int] = None,
        filler_before: int = 5,
        filler_after: int = 5,
        seed: Optional[int] = None
    ) -> Dict[str, str]:
        """生成a single 範例 with a 變數 assignment buried in 上下文.
        
        參數：
            變數: 變數 名稱 (e.g., "X"). If None, randomly chosen.
            值: 值 to assign. If None, 隨機 integer 1-20.
            filler_before: 數字 of filler 詞 before the key fact.
            filler_after: 數字 of filler 詞 after the key fact.
            種子: 隨機種子 以確保可重現性.
            
        回傳：
            字典 with 輸入, 輸出, 變數, and 值.
        """
        if seed is not None:
            random.seed(seed)
        
        # 生成variable and 值 if not provided
        if variable is None:
            variable = random.choice(['X', 'Y', 'Z', 'A', 'B', 'N', 'M'])
        
        if value is None:
            value = random.randint(1, 20)
        
        # 生成filler 文字
        before = self._generate_filler_text(filler_before, seed)
        after = self._generate_filler_text(filler_after, seed=(seed+1) if seed else None)
        
        # 建立the 上下文 with embedded fact
        context = f"{before} {variable} = {value}. {after}"
        question = f"Question: What is {variable}?"
        
        # Combined 輸入
        input_str = f"{context}\n{question}"
        
        return {
            "input": input_str,
            "output": str(value),
            "variable": variable,
            "value": value,
            "context_length": len(context.split())
        }
    
    def generate_examples(
        self,
        num_examples: int,
        filler_range: tuple = (3, 8),
        seed: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """生成multiple 範例 with varying 上下文 lengths.
        
        參數：
            num_examples: 數字 of 範例 to 生成.
            filler_range: (min, max) 詞 for filler 文字.
            種子: 隨機種子 以確保可重現性.
            
        回傳：
            列表 of 範例 dictionaries.
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for i in range(num_examples):
            filler_len = random.randint(*filler_range)
            example = self.generate_example(
                filler_before=filler_len,
                filler_after=filler_len,
                seed=(seed + i) if seed else None
            )
            examples.append(example)
        
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
        filler_range: tuple = (3, 8),
    ) -> List[Dict[str, str]]:
        """
        回傳 ICL 格式的範例.
        
        If use_generator=True, 生成 合成 範例.
        Otherwise, uses the standard BaseTask logic with stored 資料.
        
        參數：
            num_examples: 數字 of 範例 to 回傳.
            打亂: Whether to 打亂 (ignored if use_generator=True).
            種子: 隨機種子 以確保可重現性.
            新的: Whether to prefer 未使用 範例 (ignored if use_generator=True).
            filler_range: (min, max) 詞 for filler 文字 in 已生成 範例.
            
        回傳：
            列表 of 範例 dictionaries with 'input' and 'output' keys.
        """
        if self.use_generator:
            # 生成fresh 範例 即時
            return self.generate_examples(num_examples, filler_range, seed)
        else:
            # Use the standard BaseTask logic with stored 資料
            return super().get_icl_examples(
                num_examples=num_examples,
                shuffle=shuffle,
                seed=seed,
                fresh=fresh
            )
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """以完全匹配準確率評估預測."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) doesn't match "
                f"ground truth count ({len(ground_truth)})"
            )
        
        # 預處理 預測
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate 完全匹配 準確率
        correct = sum(
            1 for pred, gt in zip(processed_predictions, ground_truth)
            if pred.strip() == gt.strip()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }


def make_ignoring_context_task(
    config: TaskConfig = None,
    use_generator: bool = False,
) -> IgnoringContextTask:
    """工廠函式 to 建立一個IgnoringContextTask.
    
    參數：
        設定: 可選TaskConfig. If None, creates 預設config with 範例.
        use_generator: If True, 生成synthetic 範例 即時.
    
    回傳：
        IgnoringContextTask 實例.
    """
    if config is None:
        if use_generator:
            # Minimal 設定 for generator mode
            default_examples = [
                {
                    "input": "Some text here. X = 5. More text.\nQuestion: What is X?",
                    "output": "5"
                }
            ]
        else:
            # 靜態 範例 with varying 上下文 lengths
            default_examples = [
                {
                    "input": "The cat sat. X = 3. A dog ran.\nQuestion: What is X?",
                    "output": "3",
                    "variable": "X",
                    "value": 3
                },
                {
                    "input": "Today was nice and sunny. Y = 7. We went to the park.\nQuestion: What is Y?",
                    "output": "7",
                    "variable": "Y",
                    "value": 7
                },
                {
                    "input": "The house is big and blue. Z = 12. Birds were singing.\nQuestion: What is Z?",
                    "output": "12",
                    "variable": "Z",
                    "value": 12
                },
                {
                    "input": "A car drove by fast. N = 1. The tree is tall.\nQuestion: What is N?",
                    "output": "1",
                    "variable": "N",
                    "value": 1
                },
                {
                    "input": "Books are good to read every day. M = 15. Time goes quickly.\nQuestion: What is M?",
                    "output": "15",
                    "variable": "M",
                    "value": 15
                },
            ]
        
        config = TaskConfig(
            name="ignoring_context",
            description="Extract key information while ignoring irrelevant context",
            data_format="memory",
            in_memory_data=default_examples,
            num_demonstrations=3,
        )
    
    return IgnoringContextTask(config, use_generator=use_generator)
