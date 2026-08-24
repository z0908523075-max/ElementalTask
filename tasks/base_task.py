"""基礎 任務 類別 and 設定. When implementing a 新的 任務, please put it under 任務/實作 and inherit from BaseTask."""

import json
import csv
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path


@dataclass
class TaskConfig:
    """設定 for a 任務."""
    name: str # the 任務 名稱 
    description: Optional[str] = None # 可選description
    data_path: Optional[str] = None # 路徑 to the 主要資料檔案 (CSV, JSON, etc.) for 非記憶體內 任務
    data_format: str = "csv"  # 'csv', 'json', 'jsonl', 'memory'
    input_column: str = "input" # 名稱 of the 輸入 column in the 資料
    output_column: str = "output" # 名稱 of the 輸出 column in the 資料
    demonstrations_path: Optional[str] = None # 路徑 to 示範 檔案 (JSON, CSV, TXT)
    num_demonstrations: int = 5 # 數字 of 示範 to include in 提示. Will be max(num available, num_demonstrations)
    prompt_template: Optional[str] = None # 可選prompt template with placeholders
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For 記憶體內 資料 (when migrating from hardcoded 任務)
    in_memory_data: Optional[List[Dict[str, Any]]] = None # option to just pass 記憶體內 資料 (e.g. just a 列表)
    in_memory_demonstrations: Optional[Dict[str, List[str]]] = None


class BaseTask(ABC):
    """基礎 類別 for all 任務"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.data = None
        self.demonstrations = None
        # track which 範例 indices have been used for ICL sampling
        self._icl_used_indices = set()
        self._load_data()
        self._load_demonstrations()
    
    def _load_data(self):
        """載入main 任務 資料 from the specified 路徑 and 格式化or use 記憶體內 資料."""
        # 檢查if we have 記憶體內 資料 第一個
        if self.config.in_memory_data is not None:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
            
        # If no data_path provided, let subclass handle 資料 載入
        if not self.config.data_path:
            # Subclasses should override this method to provide their own 資料
            # If they don't override it properly, self.data will remain None
            return
            
        # Otherwise 載入from 檔案
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if self.config.data_format.lower() == 'csv':
            self.data = pd.read_csv(data_path)
        elif self.config.data_format.lower() == 'json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data = pd.DataFrame(data)
        elif self.config.data_format.lower() == 'jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            self.data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {self.config.data_format}")
    
    def _load_demonstrations(self):
        """載入demonstration 範例 from 檔案, 記憶體內 資料, or use 預設."""
        # 檢查for 記憶體內 示範 第一個
        if self.config.in_memory_demonstrations is not None:
            self.demonstrations = self.config.in_memory_demonstrations
            return
            
        # Otherwise try to 載入from 檔案
        if self.config.demonstrations_path:
            demo_path = Path(self.config.demonstrations_path)
            if demo_path.exists():
                if demo_path.suffix == '.json':
                    with open(demo_path, 'r', encoding='utf-8') as f:
                        self.demonstrations = json.load(f)
                elif demo_path.suffix == '.csv':
                    self.demonstrations = pd.read_csv(demo_path)
                else:
                    # Try to 載入as 文字 檔案 with 範例
                    with open(demo_path, 'r', encoding='utf-8') as f:
                        self.demonstrations = f.read().strip()
    @property
    def supports_icl(self) -> bool:
        """檢查if this 任務 支援 ICL 格式"""
        return hasattr(self, 'get_icl_examples') or self.config.in_memory_data is not None

    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
    ) -> List[Dict[str, str]]:
        """回傳 ICL 格式的範例 (dicts with 'input' and 'output').

        參數：
            num_examples: 數字 of 範例 to 回傳.
            shuffle: whether to 打亂 可用 範例 before selection.
            seed: 可選隨機種子 以確保可重現性.
            fresh: if True, prefer 範例 that haven't been returned before
                   (tracks indices in ``self._icl_used_indices``).
        """
        rows = self.get_split("test")
        if not rows:
            return []

        n = len(rows)
        indices = list(range(n))

        if shuffle:
            import random
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        if fresh:
            # choose indices not seen before; if insufficient, wrap around deterministically
            available = [i for i in indices if i not in self._icl_used_indices]
            if len(available) < num_examples:
                # include previously used indices to fill up
                available = available + [i for i in indices if i not in available]
            selected = available[:min(num_examples, n)]
            self._icl_used_indices.update(selected)
        else:
            selected = indices[:min(num_examples, n)]

        examples: List[Dict[str, str]] = []
        for i in selected:
            item = rows[i]
            examples.append({
                "input": item.get(self.config.input_column, ""),
                "output": str(item.get(self.config.output_column, "")),
            })
        return examples

    def reset_icl_tracking(self) -> None:
        """Reset internal tracking of used ICL 範例 so sampling starts 新的."""
        self._icl_used_indices.clear()

    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """取得data 切分 as a 列表 of dictionaries."""
        if split not in ["train", "val", "test", "all"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test', 'all']")
        
        # For now, 回傳all 資料 as test 切分
        # Subclasses can override this for proper train/val/test splits
        if split == "all" or split == "test":
            # Handle both DataFrame and 列表 資料
            if hasattr(self.data, 'to_dict'):
                return self.data.to_dict('records')
            elif isinstance(self.data, list):
                return self.data
            else:
                return list(self.data)
        else:
            # 回傳empty for train/val if not implemented
            return []
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """建立提示 for a given 實例.
        
        參數：
            實例: The 實例 to 建立提示 for
            num_shots: 數字 of in-context learning 範例 to include (預設: 5)
        
        回傳：
            The 格式化 提示 字串
        """
        prompt = ""
        
        # Add 示範 若可用
        if self.demonstrations is not None:
            prompt += self._format_demonstrations()
            prompt += "\n\n"
        # If 任務 支援 ICL 生成, use that
        elif hasattr(self, 'get_icl_examples') and num_shots > 0:
            try:
                icl_examples = self.get_icl_examples(num_examples=num_shots)
                if icl_examples:
                    prompt += self._format_icl_examples(icl_examples)
                    prompt += "\n\n"
            except Exception as e:
                # If ICL 生成 fails, continue without 範例
                pass
        
        # Add the 實例 輸入
        if self.config.prompt_template:
            prompt += self.config.prompt_template.format(**instance)
        else:
            prompt += f"Input: {instance[self.config.input_column]}\nOutput:"
        
        return prompt
    
    def _format_demonstrations(self) -> str:
        """格式化demonstration 範例 into a 字串."""
        if isinstance(self.demonstrations, str):
            return self.demonstrations
        elif isinstance(self.demonstrations, pd.DataFrame):
            demo_text = ""
            for _, row in self.demonstrations.head(self.config.num_demonstrations).iterrows():
                demo_text += f"Input: {row[self.config.input_column]}\nOutput: {row[self.config.output_column]}\n\n"
            return demo_text.strip()
        elif isinstance(self.demonstrations, list):
            demo_text = ""
            for demo in self.demonstrations[:self.config.num_demonstrations]:
                demo_text += f"Input: {demo[self.config.input_column]}\nOutput: {demo[self.config.output_column]}\n\n"
            return demo_text.strip()
        elif isinstance(self.demonstrations, dict):
            # Handle category-based 示範 (like in the notebook)
            return ""  # Will be handled by subclasses that need category-specific demos
        else:
            return ""
    
    def _format_icl_examples(self, examples: List[Dict[str, Any]]) -> str:
        """格式化ICL 範例 into a 字串.
        
        參數：
            範例: 列表 of 範例 dictionaries with 輸入/輸出 keys
            
        回傳：
            格式化 字串 of 範例
        """
        if not examples:
            return ""
        
        demo_text = "\n"
        for ex in examples:
            # Try different column 名稱
            input_val = ex.get(self.config.input_column) or ex.get('input') or ex.get('question')
            output_val = ex.get(self.config.output_column) or ex.get('output') or ex.get('answer')
            
            if input_val is not None and output_val is not None:
                demo_text += f"Input: {input_val}\nOutput: {output_val}\n\n"
        
        return demo_text.strip()
    
    @abstractmethod
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """評估預測 against 真值."""
        pass
    
    def preprocess_prediction(self, prediction: str) -> str:
        """預處理 a 模型 預測 before 評估."""
        # 預設 實作: strip whitespace and take 第一個 行
        return prediction.strip().split('\n')[0] if prediction else ""
    
    def get_ground_truth(self, split: str = "test") -> List[str]:
        """取得真值 答案 for a 切分."""
        data = self.get_split(split)
        return [item[self.config.output_column] for item in data]


class SimpleTask(BaseTask):
    """A 簡單 任務 實作 with basic 完全匹配 評估."""
    
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


# 工廠函式 for 建立中 任務 from 設定 檔案
def create_task_from_config(config_path: str, task_class: type = SimpleTask) -> BaseTask:
    """建立一個task 實例 from a 設定 檔案."""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    config = TaskConfig(**config_dict)
    return task_class(config)


# 工廠函式 for 建立中 任務 from hardcoded 資料 (for migration)
def create_task_from_data(
    name: str,
    data: List[Dict[str, Any]],
    demonstrations: Optional[Dict[str, List[str]]] = None,
    description: str = "Task created from hardcoded data",
    task_class: type = SimpleTask,
    **kwargs
) -> BaseTask:
    """建立一個task 實例 from 記憶體內 資料."""
    config = TaskConfig(
        name=name,
        description=description,
        data_format="memory",
        in_memory_data=data,
        in_memory_demonstrations=demonstrations,
        **kwargs
    )
    return task_class(config)


# 工具 function to 轉換notebook-style 資料 to standardized 格式
def convert_notebook_data_to_standard(
    csv_path: str,
    category_demonstrations: Dict[str, List[str]],
    output_csv_path: Optional[str] = None,
    output_demo_json_path: Optional[str] = None
) -> tuple[str, str]:
    """
    轉換notebook-style hardcoded 資料 to standardized CSV and JSON 檔案.
    
    參數：
        csv_path: 路徑 to the existing CSV 資料 檔案
        category_demonstrations: 字典 mapping 類別 to 示範 範例
        output_csv_path: 路徑 for 輸出 CSV (預設 to 輸入 路徑)
        output_demo_json_path: 路徑 for 示範 JSON 檔案
        
    回傳：
        Tuple of (csv_path, demo_json_path)
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Read the CSV 資料
    data = pd.read_csv(csv_path)
    
    # Set 預設output 路徑
    if output_csv_path is None:
        output_csv_path = csv_path
    if output_demo_json_path is None:
        base_path = Path(csv_path).parent
        output_demo_json_path = base_path / f"{Path(csv_path).stem}_demonstrations.json"
    
    # 儲存demonstrations to JSON
    with open(output_demo_json_path, 'w', encoding='utf-8') as f:
        json.dump(category_demonstrations, f, indent=2, ensure_ascii=False)
    
    # Ensure CSV has the right 格式化(it likely already does)
    if output_csv_path != csv_path:
        data.to_csv(output_csv_path, index=False)
    
    print(f"Converted data saved to:")
    print(f"  CSV: {output_csv_path}")
    print(f"  Demonstrations: {output_demo_json_path}")
    
    return str(output_csv_path), str(output_demo_json_path)
