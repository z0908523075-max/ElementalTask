"""Copying 用於testing induction heads.

This 任務 presents 範例 where 輸入 equals 輸出, testing the 模型's
ability to perform exact copying - a key capability of induction heads.

Two modes of 操作:

1. **靜態 mode** (use_generator=False):
   - Uses a fixed set of 範例 (預設: 10 簡單 詞/字串)
   - ICL 範例 are sampled from this 靜態 set
   - Good 以確保可重現性 and consistency
   
2. **Generator mode** (use_generator=True):
   - 生成 unlimited 隨機 字元 sequences 即時
   - Each call to get_icl_examples() produces 新的 隨機 字串
   - Configurable length range and 字元 set
   - Perfect for testing with diverse inputs without 資料 limitations

範例 usage:
    # 靜態 mode
    task = make_copying_task(use_generator=False)
    範例 = task.get_icl_examples(num_examples=5)
    
    # Generator mode with 自訂 parameters
    task = make_copying_task(use_generator=True, min_length=3, max_length=8)
    範例 = task.get_icl_examples(num_examples=10, charset="abc123", seed=42)
"""

import random
import string
from tasks.base_task import BaseTask, TaskConfig
from typing import Dict, List, Iterator


class CopyingTask(BaseTask):
    """
    A 任務 where the 輸出 is an exact copy of the 輸入.
    This tests induction head capabilities in language 模型.
    """
    TASK_NAME = "copying"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False, 
                 min_length: int = 3, max_length: int = 8):
        """
        初始化the CopyingTask.
        
        參數：
            設定: TaskConfig with 任務 settings
            use_generator: If True, 生成random 範例 即時 instead of using 靜態 資料
            min_length: Minimum length of 已生成 隨機 字串 (預設: 3)
            max_length: Maximum length of 已生成 隨機 字串 (預設: 8)
        """
        self.use_generator = use_generator
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(config)
    
    def generate_random_example(self, length: int = None, 
                               charset: str = None, 
                               seed: int = None) -> Dict[str, str]:
        """
        生成a 隨機 copying 範例.
        
        參數：
            length: Length of the 隨機 字串. If None, randomly chosen between min_length and max_length
            charset: 字元 set to use. If None, uses letters and digits
            seed: 隨機種子 以確保可重現性
            
        回傳：
            字典 with 'input' and 'output' keys (both identical)
        """
        if seed is not None:
            random.seed(seed)
        
        if charset is None:
            charset = string.ascii_letters + string.digits
        
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        random_str = ''.join(random.choices(charset, k=length))
        return {"input": random_str, "output": random_str}
    
    def generate_examples(self, num_examples: int, 
                         charset: str = None,
                         seed: int = None) -> List[Dict[str, str]]:
        """
        生成multiple 隨機 copying 範例.
        
        參數：
            num_examples: 數字 of 範例 to 生成
            charset: 字元 set to use for 生成
            seed: 隨機種子 以確保可重現性
            
        回傳：
            列表 of 範例 dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for _ in range(num_examples):
            examples.append(self.generate_random_example(charset=charset))
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: int = None,
        fresh: bool = True,
        charset: str = None,
    ) -> List[Dict[str, str]]:
        """
        回傳 ICL 格式的範例.
        
        If use_generator=True, 生成 新的 隨機 範例 each time.
        Otherwise, uses the standard BaseTask logic with stored 資料.
        
        參數：
            num_examples: 數字 of 範例 to 回傳
            shuffle: Whether to 打亂 (ignored if use_generator=True)
            seed: 隨機種子 以確保可重現性
            fresh: Whether to prefer 未使用 範例 (ignored if use_generator=True)
            charset: 字元 set for 隨機 生成 (only used if use_generator=True)
            
        回傳：
            列表 of 範例 dictionaries with 'input' and 'output' keys
        """
        if self.use_generator:
            # 生成fresh 隨機 範例 即時
            return self.generate_examples(num_examples, charset=charset, seed=seed)
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


def make_copying_task(config: TaskConfig = None, 
                      use_generator: bool = False,
                      min_length: int = 3,
                      max_length: int = 8,
                      test_size: int = 20) -> CopyingTask:
    """工廠函式 to 建立一個CopyingTask with 預設examples.
    
    參數：
        設定: 可選TaskConfig. If None, creates a 預設config with
                範例 資料 where 輸入 == 輸出.
        use_generator: If True, 生成random 範例 即時 instead of using 靜態 資料.
                      This allows for limitless unique 範例.
        min_length: Minimum length of 已生成 隨機 字串 (only used if use_generator=True)
        max_length: Maximum length of 已生成 隨機 字串 (only used if use_generator=True)
        test_size: 數字 of test 範例 to 生成(預設: 20)
    
    回傳：
        CopyingTask 實例
    """
    if config is None:
        if use_generator:
            # 生成test 範例 once at 任務 creation for deterministic 評估
            # We 建立一個temporary 實例 just to use the generate_examples method
            import random
            import string
            
            # 生成deterministic test set
            random.seed(42)
            test_examples = []
            for _ in range(test_size):
                length = random.randint(min_length, max_length)
                charset = string.ascii_letters + string.digits
                random_str = ''.join(random.choices(charset, k=length))
                test_examples.append({"input": random_str, "output": random_str})
        else:
            # 預設 範例 - 簡單 字串 that should be copied exactly
            test_examples = [
                {"input": "cat", "output": "cat"},
                {"input": "dog", "output": "dog"},
                {"input": "hello", "output": "hello"},
                {"input": "world", "output": "world"},
                {"input": "apple", "output": "apple"},
                {"input": "banana", "output": "banana"},
                {"input": "123", "output": "123"},
                {"input": "xyz", "output": "xyz"},
                {"input": "test", "output": "test"},
                {"input": "copy", "output": "copy"},
            ]
        
        config = TaskConfig(
            name="copying",
            description="Task where output exactly copies the input (induction heads)",
            data_format="memory",
            in_memory_data=test_examples,
            num_demonstrations=5,
        )
    
    return CopyingTask(config, use_generator=use_generator, 
                      min_length=min_length, max_length=max_length)
