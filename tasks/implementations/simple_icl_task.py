"""簡單 ICL 任務 實作 with category-based 示範."""

from typing import Dict, List, Any
import pandas as pd
from ..base_task import BaseTask, TaskConfig


class SimpleICLTask(BaseTask):
    """用於simple in-context learning with category-based 示範."""
    
    TASK_NAME = "simple_icl"  # 自動註冊名稱
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        
        # Define hardcoded 示範 for each 類別
        import string
        import random

        # 建立full alphabet 示範 for completeness.
        # 打亂 with a fixed 種子 so the 5 shown demos are non-consecutive,
        # which eliminates the "consecutive alphabetic shift" ambiguity that
        # arises when the 模型 always sees a->A, b->B, c->C, d->D, e->E.
        _rng = random.Random(42)
        uppercase_demos = [f"{ch.lower()} -> {ch}" for ch in string.ascii_uppercase]
        _rng.shuffle(uppercase_demos)
        lowercase_demos = [f"{ch} -> {ch.lower()}" for ch in string.ascii_uppercase]
        _rng.shuffle(lowercase_demos)

        self.category_demonstrations = {
            "uppercase": uppercase_demos,
            "lowercase": lowercase_demos,
            "first_letter": [
                "the cat went up the tree -> t",
                "elephants are cool -> e",
                "beautiful morning sunshine -> b",
                "running through the forest -> r",
                "amazing journey ahead -> a",
            ],
            "last_letter": [
                "the cat went up the tree -> e",
                "elephants are cool -> l",
                "beautiful morning sunshine -> e",
                "running through the forest -> t",
                "amazing journey ahead -> d",
            ],
            "translate_eng_fr": [
                "hello -> bonjour",
                "goodbye -> au revoir",
                "thank you -> merci",
                "please -> s'il vous plaît",
                "yes -> oui",
                "cat -> chat",
                "dog -> chien",
                "house -> maison",
                "car -> voiture",
                "book -> livre",
                "flower -> fleur",
                "pen -> stylo",
                "pencil -> crayon",
            ],
            "translate_fr_eng": [
                "bonjour -> hello",
                "au revoir -> goodbye",
                "merci -> thank you",
                "s'il vous plaît -> please",
                "oui -> yes",
                "chat -> cat",
                "chien -> dog",
                "maison -> house",
                "voiture -> car",
                "livre -> book",
                "fleur -> flower",
                "stylo -> pen",
                "crayon -> pencil",
            ],
            "translate_eng_sp": [
                "hello -> hola",
                "goodbye -> adiós",
                "thank you -> gracias",
                "please -> por favor",
                "yes -> sí",
                "cat -> gato",
                "dog -> perro",
                "house -> casa",
                "car -> coche",
                "book -> libro",
                "flower -> flor",
                "pen -> bolígrafo",
                "pencil -> lápiz",
                "flower -> flor",
            ],
            "translate_sp_eng": [
                "hola -> hello",
                "adiós -> goodbye",
                "gracias -> thank you",
                "por favor -> please",
                "sí -> yes",
                "gato -> cat",
                "perro -> dog",
                "casa -> house",
                "coche -> car",
                "libro -> book",
                "flor -> flower",
                "bolígrafo -> pen",
                "lápiz -> pencil",
                "flor -> flower"
            ],
            "present_to_gerund": [
                "run -> running", "swim -> swimming", "walk -> walking",
                "jump -> jumping", "dance -> dancing", "sing -> singing",
                "read -> reading", "write -> writing", "sleep -> sleeping",
            ],
            "singular_to_plural": [
                "cat -> cats", "dog -> dogs", "book -> books", "car -> cars",
                "house -> houses", "child -> children", "mouse -> mice",
            ],
            "country_to_capital": [
                "France -> Paris", "Germany -> Berlin", "Italy -> Rome",
                "Spain -> Madrid", "United Kingdom -> London", "Japan -> Tokyo",
                "Canada -> Ottawa", "Australia -> Canberra",
            ],
            "country_to_currency": [
                "France -> Euro", "United States -> Dollar", "United Kingdom -> Pound",
                "Japan -> Yen", "Canada -> Canadian Dollar", "Australia -> Australian Dollar",
                "India -> Rupee", "China -> Yuan",
            ],
        }
    
    def _load_data(self):
        """載入data from CSV 檔案."""
        import pandas as pd
        
        if self.config.data_path:
            self.data = pd.read_csv(self.config.data_path)
        else:
            # If no 資料 路徑, 建立empty DataFrame
            self.data = pd.DataFrame(columns=[self.config.input_column, self.config.output_column])
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """建立提示 with category-specific 示範.
        
        參數：
            實例: The 實例 to 建立提示 for
            num_shots: 數字 of in-context learning 範例 (預設: 5)
        
        回傳：
            格式化 提示 字串
        """
        category = instance.get('category_name', '')
        
        # 檢查if we have 示範 in the 設定 (from 記憶體內 資料)
        if (self.config.in_memory_demonstrations and 
            isinstance(self.config.in_memory_demonstrations, dict)):
            demos = self.config.in_memory_demonstrations.get(category, [])
        else:
            # Fallback to hardcoded 示範
            demos = self.category_demonstrations.get(category, [])
        
        # Exclude 示範 that match the 當前 實例 輸入 to avoid leaking the test item
        instance_input = instance.get(self.config.input_column, '')
        filtered_demos = []
        for d in demos:
            # Each demo may be 格式化 like '輸入 -> 輸出'
            demo_input = d.split('->', 1)[0].strip()
            if demo_input != instance_input:
                filtered_demos.append(d)

        # Limit to num_shots 若有提供
        if filtered_demos and num_shots > 0:
            demos = filtered_demos[:num_shots]
        else:
            demos = filtered_demos
        
        prompt = ""
        if demos:
            for demo in demos:
                prompt += f"{demo}\n"
        
        # Add the 當前 問題
        prompt += f"{instance[self.config.input_column]} ->"
        
        return prompt
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """以完全匹配準確率評估預測, including per-category metrics."""
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # 預處理 預測
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Overall 準確率
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.lower().strip() == gt.lower().strip())
        accuracy = correct / len(ground_truth)
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Per-category 準確率
        category_stats = {}
        for pred, gt, item in zip(processed_predictions, ground_truth, task_data):
            category = item.get("category_name", "unknown")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            
            category_stats[category]["total"] += 1
            if pred.lower().strip() == gt.lower().strip():
                category_stats[category]["correct"] += 1
        
        # Calculate per-category 準確率
        for category, stats in category_stats.items():
            results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
            results[f"correct_{category}"] = stats["correct"]
            results[f"total_{category}"] = stats["total"]
        
        return results
