"""字串 類比 任務 based on Melanie Mitchell's work.
The 預設ICL 範例 are taken from: https://melaniemitchell.me/ExplorationsContent/analogy-problems.html

格式: source -> target, query -> ?
任務: Identify the transformation from source to target, then apply it to query.

範例: abc -> abd, ijk -> ?
答案: ijl (replace 最後一個 letter with successor)
"""

import random
import string
from typing import Dict, List, Optional, Tuple
from tasks.base_task import BaseTask, TaskConfig


class StringAnalogyTask(BaseTask):
    """
    字串 類比 用於testing 類比 推理.
    
    格式: source -> target, query -> ?
    The 模型 must identify the transformation and apply it to the query.
    """
    TASK_NAME = "string_analogy"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False):
        """
        初始化StringAnalogyTask.
        
        參數：
            設定: TaskConfig with 任務 settings
            use_generator: If True, 生成synthetic 範例 for ICL
        """
        self.use_generator = use_generator
        super().__init__(config)
    
    def _successor(self, char: str) -> str:
        """取得next letter in alphabet (wraps z->a)."""
        if char.isalpha():
            if char == 'z':
                return 'a'
            elif char == 'Z':
                return 'A'
            return chr(ord(char) + 1)
        return char
    
    def _predecessor(self, char: str) -> str:
        """取得previous letter in alphabet (wraps a->z)."""
        if char.isalpha():
            if char == 'a':
                return 'z'
            elif char == 'A':
                return 'Z'
            return chr(ord(char) - 1)
        return char
    
    def _apply_transformation(self, transformation: str, query: str) -> str:
        """Apply a transformation rule to a query 字串."""
        
        if transformation == "successor_last":
            # Replace 最後一個 字元 with successor
            if not query:
                return query
            return query[:-1] + self._successor(query[-1])
        
        elif transformation == "successor_all":
            # Replace all 字元 with successors
            return ''.join(self._successor(c) for c in query)
        
        elif transformation == "predecessor_last":
            # Replace 最後一個 字元 with predecessor
            if not query:
                return query
            return query[:-1] + self._predecessor(query[-1])
        
        elif transformation == "reverse":
            # Reverse the 字串
            return query[::-1]
        
        elif transformation == "double_string":
            # Double the entire 字串
            return query + query
        
        elif transformation == "double_each":
            # Double each 字元
            return ''.join(c * 2 for c in query)
        
        elif transformation == "delete_middle":
            # Delete middle character(s)
            if len(query) <= 1:
                return query
            mid = len(query) // 2
            if len(query) % 2 == 1:
                return query[:mid] + query[mid+1:]
            else:
                return query[:mid-1] + query[mid+1:]
        
        elif transformation == "delete_first":
            # Delete 第一個 字元
            return query[1:] if len(query) > 0 else query
        
        elif transformation == "delete_last":
            # Delete 最後一個 字元
            return query[:-1] if len(query) > 0 else query
        
        elif transformation == "append_successor":
            # Append the successor of the 最後一個 字元
            if not query:
                return query
            return query + self._successor(query[-1])
        
        elif transformation == "insert_x_middle":
            # Insert 'x' in the middle
            mid = len(query) // 2
            return query[:mid] + 'x' + query[mid:]
        
        elif transformation == "swap_halves":
            # Swap 第一個 and second half
            mid = len(query) // 2
            return query[mid:] + query[:mid]
        
        elif transformation == "delete_duplicates":
            # Remove consecutive duplicates
            if not query:
                return query
            result = [query[0]]
            for c in query[1:]:
                if c != result[-1]:
                    result.append(c)
            return ''.join(result)
        
        else:
            # Unknown transformation, 回傳query unchanged
            return query
    
    def generate_analogy_example(
        self, 
        transformation: str,
        source_length: int = 3,
        seed: Optional[int] = None
    ) -> Dict[str, str]:
        """生成a 合成 類比 範例 with a given transformation.
        
        For ICL 格式, this 生成 just a single 輸入->輸出 pair showing
        the transformation. Multiple 範例 of the 相同 transformation type
        should be used together for proper ICL 示範.
        
        參數：
            transformation: Type of transformation to apply
            source_length: Length of the source 字串
            種子: 隨機種子 以確保可重現性
            
        回傳：
            字典 with 輸入, 輸出, and transformation fields
        """
        if seed is not None:
            random.seed(seed)
        
        # 生成source/query 字串 and apply the 相同 transformation to both,
        # so 範例 match the 任務's 類比 格式.
        source = ''.join(random.choices(string.ascii_lowercase, k=source_length))
        query_length = random.randint(2, 5)
        query = ''.join(random.choices(string.ascii_lowercase, k=query_length))

        target = self._apply_transformation(transformation, source)
        answer = self._apply_transformation(transformation, query)

        return {
            "input": f"{source} -> {target}, {query} -> ?",
            "output": answer,
            "source": source,
            "target": target,
            "query": query,
            "answer": answer,
            "transformation": transformation,
        }
    
    def generate_examples(
        self,
        num_examples: int,
        transformations: Optional[List[str]] = None,
        seed: Optional[int] = None,
        same_transformation: bool = True
    ) -> List[Dict[str, str]]:
        """生成multiple 類比 範例.
        
        參數：
            num_examples: 數字 of 範例 to 生成
            transformations: 列表 of transformation types to use (if None, use all)
            種子: 隨機種子 以確保可重現性
            same_transformation: If True, all 範例 use the 相同 transformation type
                                (proper ICL 格式). If False, mix different transformations.
            
        回傳：
            列表 of 範例 dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        if transformations is None:
            transformations = [
                "successor_last", "successor_all", "predecessor_last",
                "reverse", "double_string", "double_each",
                "delete_middle", "delete_first", "delete_last",
                "append_successor", "swap_halves"
            ]
        
        examples = []
        
        if same_transformation:
            # Pick one transformation and 生成all 範例 with it
            trans = random.choice(transformations)
            for i in range(num_examples):
                length = random.randint(2, 4)
                example = self.generate_analogy_example(trans, length, seed=seed+i if seed else None)
                examples.append(example)
        else:
            # Mix different transformations
            for i in range(num_examples):
                trans = random.choice(transformations)
                length = random.randint(2, 4)
                example = self.generate_analogy_example(trans, length, seed=seed+i if seed else None)
                examples.append(example)
        
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
        transformations: Optional[List[str]] = None,
        same_transformation: bool = True,
    ) -> List[Dict[str, str]]:
        """
        回傳 ICL 格式的範例.
        
        If use_generator=True, 生成 合成 範例 with the 相同 transformation.
        Otherwise, uses the standard BaseTask logic with stored 資料.
        
        參數：
            num_examples: 數字 of 範例 to 回傳
            打亂: Whether to 打亂 (ignored if use_generator=True)
            種子: 隨機種子 以確保可重現性
            新的: Whether to prefer 未使用 範例 (ignored if use_generator=True)
            transformations: 列表 of transformation types for generator mode
            same_transformation: If True, all 已生成 範例 use the 相同 transformation
                                (recommended for proper ICL 格式)
            
        回傳：
            列表 of 範例 dictionaries with 'input' and 'output' keys
        """
        if self.use_generator:
            # 生成fresh 範例 即時
            # Use same_transformation=True by 預設for proper ICL
            examples = self.generate_examples(
                num_examples, 
                transformations, 
                seed,
                same_transformation=same_transformation
            )
            return examples
        else:
            # Use the standard BaseTask logic with stored 資料
            return super().get_icl_examples(
                num_examples=num_examples,
                shuffle=shuffle,
                seed=seed,
                fresh=fresh
            )

    def build_prompt(self, instance: Dict[str, str], num_shots: int = 5) -> str:
        """建立提示 with analogy-formatted 示範.

        If transformation metadata is 可用, prefer 示範 that use the
        相同 transformation to keep few-shot 上下文 coherent.
        """
        prompt = ""

        if num_shots > 0:
            transformation = instance.get("transformation")

            if self.use_generator and transformation:
                icl_examples = self.get_icl_examples(
                    num_examples=num_shots,
                    seed=42,
                    transformations=[transformation],
                    same_transformation=True,
                )
            else:
                rows = self.get_split("test")
                if transformation:
                    rows = [r for r in rows if r.get("transformation") == transformation]

                # Avoid leaking the exact query row into 示範.
                rows = [r for r in rows if r.get("input") != instance.get("input")]

                rng = random.Random(42)
                rng.shuffle(rows)
                icl_examples = rows[:num_shots]

            if icl_examples:
                prompt += self._format_icl_examples(icl_examples)
                prompt += "\n\n"

        prompt += f"Input: {instance.get(self.config.input_column, '')}\nOutput:"
        return prompt
    
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
            if pred.strip().lower() == gt.strip().lower()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Add per-transformation 準確率 若可用
        data = self.get_split(split)
        if any("transformation" in item for item in data):
            trans_results = {}
            for pred, gt, item in zip(processed_predictions, ground_truth, data):
                trans = item.get("transformation", "unknown")
                if trans not in trans_results:
                    trans_results[trans] = {"correct": 0, "total": 0}
                
                trans_results[trans]["total"] += 1
                if pred.strip().lower() == gt.strip().lower():
                    trans_results[trans]["correct"] += 1
            
            # Calculate per-transformation 準確率
            for trans, stats in trans_results.items():
                results[f"accuracy_{trans}"] = stats["correct"] / stats["total"]
        
        return results


def make_string_analogy_task(
    config: TaskConfig = None,
    use_generator: bool = False,
    use_fixed_examples: bool = True
) -> StringAnalogyTask:
    """工廠函式 to 建立一個StringAnalogyTask.
    
    參數：
        設定: 可選TaskConfig. If None, creates 預設config.
        use_generator: If True, 生成synthetic 範例 for ICL.
        use_fixed_examples: If True and 設定 is None, use Mitchell's curated 範例.
    
    回傳：
        StringAnalogyTask 實例
    """
    if config is None:
        if use_fixed_examples:
            # Melanie Mitchell's curated 字串 類比 範例
            # 格式: source -> target, query -> 答案
            mitchell_data = [
                {"input": "abc -> abd, ijk -> ?", "output": "ijl", "source": "abc", "target": "abd", "query": "ijk", "answer": "ijl", "transformation": "successor_last"},
                {"input": "abc -> abd, xyz -> ?", "output": "xya", "source": "abc", "target": "abd", "query": "xyz", "answer": "xya", "transformation": "successor_last"},
                {"input": "abc -> abd, mrrjjj -> ?", "output": "mrrjjk", "source": "abc", "target": "abd", "query": "mrrjjj", "answer": "mrrjjk", "transformation": "successor_last"},
                {"input": "abc -> ac, pqr -> ?", "output": "pr", "source": "abc", "target": "ac", "query": "pqr", "answer": "pr", "transformation": "delete_middle"},
                {"input": "abc -> abcd, pqr -> ?", "output": "pqrs", "source": "abc", "target": "abcd", "query": "pqr", "answer": "pqrs", "transformation": "append_successor"},
                {"input": "a -> b, c -> ?", "output": "d", "source": "a", "target": "b", "query": "c", "answer": "d", "transformation": "successor_all"},
                {"input": "a -> aa, b -> ?", "output": "bb", "source": "a", "target": "aa", "query": "b", "answer": "bb", "transformation": "double_string"},
                {"input": "a -> ab, z -> ?", "output": "za", "source": "a", "target": "ab", "query": "z", "answer": "za", "transformation": "append_successor"},
                {"input": "pqr -> rqp, abc -> ?", "output": "cba", "source": "pqr", "target": "rqp", "query": "abc", "answer": "cba", "transformation": "reverse"},
                {"input": "abc -> abd, abcabc -> ?", "output": "abdabd", "source": "abc", "target": "abd", "query": "abcabc", "answer": "abdabd", "transformation": "successor_last"},
            ]
            
            config = TaskConfig(
                name="string_analogy",
                description="String analogy task (Melanie Mitchell): identify transformation and apply to query",
                data_format="memory",
                in_memory_data=mitchell_data,
                input_column="input",
                output_column="output",
                num_demonstrations=5,
            )
        else:
            # Minimal 預設for generator mode
            config = TaskConfig(
                name="string_analogy",
                description="String analogy task with generated examples",
                data_format="memory",
                in_memory_data=[{"input": "abc -> abd, ijk -> ?", "output": "ijl"}],
                num_demonstrations=5,
            )
    
    return StringAnalogyTask(config, use_generator=use_generator)
