"""間接受詞識別 (代詞消解) 用於identifying 正確 referents in 上下文 using mib-bench/ioi 資料集."""

from typing import Dict, List, Any
import re

from tasks.base_task import BaseTask, TaskConfig
from datasets import load_dataset


class IOITask(BaseTask):
    """用於間接受詞識別 using IOI 資料集 from mib-bench/ioi."""

    TASK_NAME = "ioi_task"  # 自動發現鍵
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
    pass
    
    def _load_data(self):
        """載入IOI (間接受詞識別) 資料集 from mib-bench."""
        self.data = load_dataset("mib-bench/ioi")

    def get_icl_examples(self, num_examples: int = 10, shuffle: bool = True, seed: int = None, fresh: bool = True) -> List[Dict[str, str]]:
        """回傳ICL 範例 for IOI-style 資料 using the test 切分."""
        data = self.get_split("test")
        if not data:
            return []

        indices = list(range(len(data)))
        if shuffle:
            import random
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        selected = indices[:min(num_examples, len(indices))]
        examples = []
        for i in selected:
            example = data[i]
            prompt = example.get('prompt', '')
            choices = example.get('choices', [])
            answer_index = example.get('answer_index', 0)
            output = choices[answer_index] if choices and 0 <= answer_index < len(choices) else example.get('answer', '')
            examples.append({"input": prompt, "output": str(output)})
        return examples

    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """取得data 切分 for 評估."""
        if split in self.data:
            split_data = self.data[split]
            # 轉換to 列表 of dictionaries
            return [dict(example) for example in split_data]
        elif hasattr(self.data, 'to_dict'):
            return self.data.to_dict('records')
        return self.data if isinstance(self.data, list) else []
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """建立提示 for 代詞消解 problems.
        
        參數：
            實例: The 實例 to 建立提示 for
            num_shots: 數字 of in-context learning 範例 (預設: 5)
        
        回傳：
            格式化 提示 字串
        """
        prompt_text = instance.get('prompt', '')

        sections = [
            "Complete the following sentence by identifying who should be referenced.",
            "Answer with only the name of the person who should be referenced.",
        ]

        if num_shots > 0:
            icl_examples = self.get_icl_examples(
                num_examples=num_shots,
                shuffle=True,
                seed=42,
                fresh=False,
            )
            if icl_examples:
                demo_lines = []
                for ex in icl_examples:
                    demo_lines.append(f"Sentence: {ex.get('input', '')}")
                    demo_lines.append(f"Answer: {ex.get('output', '')}")
                    demo_lines.append("")
                sections.append("\n".join(demo_lines).strip())
                sections.append("Now answer the next one.")

        sections.append(f"Sentence: {prompt_text}")
        sections.append("Answer:")
        return "\n\n".join(sections)
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """評估間接受詞識別 預測."""
        data = self.get_split(split)
        
        if len(predictions) != len(data):
            return {
                'accuracy': 0.0,
                'error': f'Prediction count ({len(predictions)}) does not match data count ({len(data)})'
            }
        
        correct = 0
        total = len(predictions)
        detailed_results = []
        
        for pred, example in zip(predictions, data):
            # 取得correct 答案 from the choices and answer_index
            choices = example.get('choices', [])
            answer_index = example.get('answer_index', 0)
            
            if choices and 0 <= answer_index < len(choices):
                expected = choices[answer_index]
            else:
                # Fallback to other possible 答案 fields
                expected = example.get('answer', example.get('correct_answer', ''))
            
            # 擷取 名稱 from 預測 and expected 答案
            pred_name = self._extract_name(pred)
            expected_name = self._extract_name(expected)
            
            # 檢查if 預測 matches expected 答案
            is_correct = (pred_name is not None and 
                         expected_name is not None and 
                         pred_name.lower() == expected_name.lower())
            
            if is_correct:
                correct += 1
            
            detailed_results.append({
                'prompt': example.get('prompt', ''),
                'prediction': pred.strip(),
                'expected': expected,
                'pred_name': pred_name,
                'expected_name': expected_name,
                'correct': is_correct,
                'choices': choices,
                'answer_index': answer_index
            })
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'detailed_results': detailed_results
        }
    
    def _extract_name(self, text: str) -> str:
        """擷取 the 名稱 from 文字, handling various response formats."""
        if not text:
            return None
        
        text = text.strip()
        
        # Common patterns in responses
        patterns = [
            r'^([A-Z][a-z]+)(?:\s|$)',  # 名稱 at start
            r'(?:is|answer|name)?\s*:?\s*([A-Z][a-z]+)',  # After common prefixes
            r'([A-Z][a-z]+)(?:\s*\.?\s*$)',  # 名稱 at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: look for any capitalized 詞 that could be a 名稱
        words = text.split()
        for word in words:
            # Remove punctuation and 檢查if it's a likely 名稱
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and clean_word.isalpha():
                # Additional check: common English 名稱 are typically 3+ 字元
                if len(clean_word) >= 3:
                    return clean_word
        
        return None


def create_ioi_task(
    name: str = "ioi_task"
) -> IOITask:
    """建立一個IOITask 實例 using mib-bench/ioi 資料集."""
    config = TaskConfig(
        name=name,
        description="Indirect Object Identification (IOI) evaluation task using mib-bench/ioi dataset",
        data_format="huggingface",
        data_path="mib-bench/ioi",  # HuggingFace 資料集 名稱
        input_column="prompt",  # The incomplete sentence
        output_column="choices",  # 列表 of choices, use with answer_index
        evaluation_metrics=["accuracy"],
        metadata={
            "task_type": "pronoun_resolution", 
            "requires_reasoning": True,
            "dataset_source": "mib-bench/ioi",
            "task_description": "Indirect Object Identification (IOI) - complete sentences by identifying the correct indirect object"
        }
    )

    return IOITask(config)