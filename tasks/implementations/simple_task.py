"""簡單 任務 實作 with basic 完全匹配 評估."""

from typing import Dict, List, Any
from ..base_task import BaseTask, TaskConfig


class SimpleTask(BaseTask):
    """A 簡單 任務 實作 with basic 完全匹配 評估."""
    
    TASK_NAME = "simple"  # 自動註冊名稱
    
    def _load_data(self):
        """載入data from CSV 檔案."""
        import pandas as pd
        
        if self.config.data_path:
            self.data = pd.read_csv(self.config.data_path)
        else:
            # If no 資料 路徑, 建立empty DataFrame
            self.data = pd.DataFrame(columns=[self.config.input_column, self.config.output_column])
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """以完全匹配準確率評估預測."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # 預處理 預測
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate 完全匹配 準確率
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.lower().strip() == gt.lower().strip())
        accuracy = correct / len(ground_truth)
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Add per-category 準確率 若可用
        data = self.get_split(split)
        if any("category" in item for item in data):
            category_results = {}
            for pred, gt, item in zip(processed_predictions, ground_truth, data):
                category = item.get("category", "unknown")
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                
                category_results[category]["total"] += 1
                if pred.lower().strip() == gt.lower().strip():
                    category_results[category]["correct"] += 1
            
            # Calculate per-category 準確率
            for category, stats in category_results.items():
                results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
        
        return results
