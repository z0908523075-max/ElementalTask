"""範例 of a 自訂 任務 with 自動 發現."""

from ..base_task import BaseTask, TaskConfig
import re
from typing import List, Dict


class MathTask(BaseTask):
    """A 用於simple 數學 problems."""
    
    TASK_NAME = "math"  # This is all that's needed for 自動 註冊!
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
    
    def _load_data(self):
        """生成synthetic 數學 problems for testing."""
        import pandas as pd
        import random
        
        random.seed(42)  # 以確保可重現性
        problems = []
        
        # 生成20 簡單 算術 problems
        for _ in range(20):
            a, b = random.randint(1, 20), random.randint(1, 20)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            else:  # multiplication
                answer = a * b
            
            problems.append({
                self.config.input_column: f"{a} {op} {b}",
                self.config.output_column: str(answer)
            })
        
        self.data = pd.DataFrame(problems)

    
    def get_icl_examples(self, num_examples: int = 10, shuffle: bool = True, seed: int = None, fresh: bool = True) -> List[Dict[str, str]]:
        """生成simple 算術 範例 for ICL.

        Note: `fresh` is ignored for 合成/已生成 任務 (there's no stable 資料集 indices to track).
        """
        import random
        if seed is not None:
            random.seed(seed)

        examples = []
        for _ in range(num_examples):
            a, b = random.randint(1, 10), random.randint(1, 10)
            examples.append({"input": f"{a} + {b}", "output": str(a + b)})

        if shuffle:
            # 打亂 in-place for diversity
            random.shuffle(examples)

        return examples[:num_examples]
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """評估math 預測."""
        correct = 0
        total = len(predictions)
        
        if total == 0:
            return {"accuracy": 0.0}
        
        # 取得targets for this 切分
        data_split = self.data if split == "test" else self.data  # For now, use 相同 資料
        targets = data_split[self.config.output_column].tolist()
        
        if len(predictions) != len(targets):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")
        
        detailed_results = []
        for pred, target in zip(predictions, targets):
            result = self.evaluate_response(pred, str(target))
            if result["correct"]:
                correct += 1
            detailed_results.append(result)
        
        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
            "detailed_results": detailed_results
        }
    
    def evaluate_response(self, response: str, target: str) -> dict:
        """評估a 數學 response with 數字 擷取."""
        
        def extract_number(text: str) -> float:
            """擷取 the 第一個 數字 from 文字."""
            # Look for 數字 (including decimals)
            match = re.search(r'-?\d+\.?\d*', text.strip())
            if match:
                return float(match.group())
            return None
        
        response_num = extract_number(response)
        target_num = extract_number(target)
        
        if response_num is None or target_num is None:
            return {
                "correct": False,
                "error": "Could not extract numbers",
                "response_extracted": response_num,
                "target_extracted": target_num
            }
        
        # 檢查if they're equal (with small tolerance for floating point)
        correct = abs(response_num - target_num) < 0.001
        
        return {
            "correct": correct,
            "response_extracted": response_num,
            "target_extracted": target_num,
            "original_response": response,
            "original_target": target
        }
    
    def generate_prompt(self, row: dict, demonstrations: list = None) -> str:
        """生成a 提示 for 數學 problems."""
        prompt = "Solve the following math problem:\n\n"
        
        if demonstrations:
            prompt += "Here are some examples:\n"
            for demo in demonstrations:
                prompt += f"Problem: {demo['input']}\nAnswer: {demo['target']}\n\n"
            prompt += "Now solve this problem:\n"
        
        prompt += f"Problem: {row[self.config.input_column]}\nAnswer:"
        return prompt
