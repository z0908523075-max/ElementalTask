"""TextFRCT 任務 實作 that integrates with the existing 資料集 utilities."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import random
import pandas as pd

from ..base_task import BaseTask, TaskConfig


class TextFRCTTask(BaseTask):
        """任務 wrapper for the TextFRCT 資料集."""
        
        TASK_NAME = "textfrct"  # 自動註冊名稱
        
        def __init__(self, config: TaskConfig, skip_subjective: bool = False, categories: Optional[List[str]] = None):
            self.skip_subjective = skip_subjective
            self.categories = categories
            super().__init__(config)
        
        def _load_data(self):
            """載入TextFRCT 資料 and optionally 篩選 by 類別 and subjective 任務."""
            data = pd.read_csv(self.config.data_path)
            
            # 篩選 by 類別 if specified
            if self.categories:
                data = data[data['category_id'].isin(self.categories)]
                print(f"Filtered to categories {self.categories}: {len(data)} examples")
            
            # 篩選 out subjective 任務 if requested
            if self.skip_subjective:
                subjective_mask = data['answer'].astype(str).str.contains('<LLMEval>', na=False)
                data = data[~subjective_mask]
                print(f"Filtered out {subjective_mask.sum()} subjective tasks, {len(data)} objective tasks remaining")
            
            # Fill NaN 值 to avoid Arrow conversion issues
            # Replace NaN with empty 字串 for 字串 columns
            data = data.fillna('')
            
            # 轉換to 列表 of dictionaries and assign to self.data
            self.data = data.to_dict('records')
        
        def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
            """取得data 切分 for TextFRCT."""
            return self.data

    # use BaseTask.get_icl_examples for standard TextFRCT records (input_column/output_column configured)
        
        def _format_category_prompt(self, instance: Dict[str, Any]) -> str:
            """格式化a single TextFRCT 實例 as a query-style 提示 block.

            提示 formats (BEFORE → AFTER for key fixes):

            RG1/RG2/RG3 — 多選題 算術/推理
              BEFORE: "Solve this problem: {q}\\nAnswer:"
              AFTER: "Solve this problem: {q}\\n\\nA. 4:30\\nB. 5:00\\n...\\nAnswer (letter):"

            MA2/MA3 — 物件/名稱 ↔ 數字 查找 (table in `additional`)
              BEFORE: "任務: ...\\nQuestion: coat\\nAnswer:"  (table never shown)
              AFTER: "tree: 58\\nfloor: 29\\n...\\n\\nQuestion: What 數字 corresponds to 'coat'?\\nAnswer:"

            RL1 — 無意義三段論 (答案 G=Good/P=Poor logic)
              BEFORE: "任務: ...\\nQuestion: {syllogism}\\nAnswer:"
              AFTER: "Does the following syllogism follow logically (regardless of whether the premises are true)?\\n{syllogism}\\nAnswer G if the logic is 有效, P if it is not. 答案 (G or P):"

            RL3/RL4, I1/I2 — inference / 多選題 with numbered choices
              BEFORE: "任務: ...\\nQuestion: {q}\\nAnswer:"  (choices never shown)
              AFTER: "Statement: {q}\\n1. ...\\n2. ...\\n\\nAnswer (數字):"

            V1/V2/V3 — 詞彙 多選題 (was mostly 正確 already)
              No change to 提示; scoring now extracts 第一個 行.
            """
            category = instance['category_id']
            question = instance['question']
            category_name = instance.get('category_name', category)
            choices_raw = instance.get('choice', '') or ''
            additional = instance.get('additional', '') or ''
            question_text = str(question).replace('<br>', '\n').strip()

            if category.startswith('CV'):  # Convergent Visual
                if category == 'CV1':  # Scrambled 詞
                    return f"Unscramble each group of letters to form a common English word. Use all the letters in each group. Respond with only the word.\n\nInput: {question}\nOutput:"
                elif category == 'CV2':  # Hidden 詞
                    return f"Find all the hidden words in the following string of letters. Words are spelled forwards and are at least 4 letters long. List them separated by semicolons.\n\nInput: {question}\nOutput:"
                elif category == 'CV3':  # Incomplete 詞
                    return f"Complete the word by filling in the missing letters.\n\nInput: {question}\nOutput:"

            elif category.startswith('FA'):  # Fluent Associational
                if category == 'FA1':  # Controlled Association
                    return f"List words that are related to or associated with '{question}'. Separate multiple answers with semicolons."
                elif category == 'FA2':  # Opposites
                    return f"List words that have the opposite meaning of '{question}'. Separate multiple answers with semicolons."

            elif category.startswith('RG'):  # 算術/推理 — multiple choice, 答案 is a letter A-E
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if choices:
                    letters = 'ABCDE'
                    choice_text = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
                    if category == 'RG3':
                        return (
                            f"Identify which arithmetic operation(s) are needed to solve this problem. "
                            f"Do not compute the final numeric result.\n\n"
                            f"Problem: {question_text}\n\n{choice_text}\n\nAnswer (letter):"
                        )
                    return f"Solve this problem: {question}\n\n{choice_text}\n\nAnswer (letter):"
                return f"Solve this problem: {question}\nAnswer:"

            elif category.startswith('MA'):  # 記憶/查找 — table in `additional`, 答案 is a 數字
                if additional:
                    table = additional.replace('<br>', '\n').strip()
                    if category == 'MA3':
                        return (
                            "FIRST AND LAST NAMES TEST\n"
                            "Learn and use first/last name pairings from the list below. "
                            "Given a last name, return the matching first name only.\n\n"
                            f"{table}\n\nLast name: {question_text}\nFirst name:"
                        )
                    return f"{table}\n\nQuestion: What number corresponds to '{question_text}'?\nAnswer:"
                if category == 'MA3':
                    return f"Last name: {question_text}\nFirst name:"
                return f"Question: What number corresponds to '{question_text}'?\nAnswer:"

            elif category == 'RL1':  # 無意義三段論 — 答案 is G (good/有效) or P (poor/無效)
                return (f"Does the following syllogism follow logically, regardless of whether "
                        f"the premises are true?\n\n{question}\n\n"
                        f"Answer G if the logic is valid, P if it is not.\nAnswer (G or P):")

            elif category.startswith('RL') or category.startswith('I'):  # Inference — numbered choices, 答案 is 1-5
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if category == 'I1' and choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    return (
                        "Four options follow one letter-pattern rule and one does not. "
                        "Pick the option that does NOT fit the same pattern.\n\n"
                        f"{choice_text}\n\nAnswer (number):"
                    )

                if category == 'I2':
                    return (
                        "Each row marks one location with an 'x'. Use the pattern across rows to determine "
                        "which numbered position (1-5) is correct.\n\n"
                        f"{question_text}\n\nAnswer (number):"
                    )

                if category == 'RL4' and choices:
                    reference = additional.replace('<br>', '\n').strip() if additional else ''
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    if reference:
                        return (
                            "DECIPHERING LANGUAGES\n"
                            "Reason across the language fragments below to infer how the ancient language maps "
                            "to the target language, then choose the best translation for the query.\n\n"
                            f"Known pairs:\n{reference}\n\n"
                            f"Query: {question_text}\n\n{choice_text}\n\nAnswer (number):"
                        )
                    return (
                        "Choose the best translation for the query from the numbered options.\n\n"
                        f"Query: {question_text}\n\n{choice_text}\n\nAnswer (number):"
                    )

                if choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    return f"Statement: {question_text}\n\nWhich conclusion follows?\n{choice_text}\n\nAnswer (number):"
                return f"Task: {category_name}\nStatement: {question}\nAnswer:"

            elif category.startswith('V'):  # 詞彙 — numbered choices, 答案 is 1-5
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    level_note = {
                        'V1': 'V1 (easier)',
                        'V2': 'V2',
                        'V3': 'V3',
                        'V4': 'V4',
                        'V5': 'V5 (harder)',
                    }.get(category, category)
                    return (
                        f"VOCABULARY TEST - {level_note}\n"
                        "Test your knowledge of word meanings.\n"
                        f"Choose the best definition for '{question_text}'. Respond with only the option number.\n\n"
                        f"{choice_text}\n\nAnswer (number):"
                    )
                return f"What does '{question}' mean?"

            # 預設 格式化for other 類別
            return f"Task: {category_name}\nQuestion: {question}\nAnswer:"

        def _with_answer(self, prompt_block: str, answer: str) -> str:
            """Attach a gold 答案 to a query-style 提示 block for 示範."""
            block = prompt_block.rstrip()
            answer_text = str(answer).strip()

            if block.endswith(":"):
                return f"{block} {answer_text}"
            return f"{block}\nAnswer: {answer_text}"

        def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
            """建立提示 with same-category few-shot 範例 before the query."""
            sections: List[str] = []

            if num_shots > 0:
                category = instance.get('category_id')
                question = instance.get('question')
                candidates = [
                    ex for ex in self.data
                    if ex.get('category_id') == category and ex.get('question') != question
                ]
                rng = random.Random(42)
                rng.shuffle(candidates)

                for demo in candidates[:num_shots]:
                    demo_block = self._format_category_prompt(demo)
                    sections.append(self._with_answer(demo_block, demo.get('answer', '')))

            sections.append(self._format_category_prompt(instance))
            return "\n\n".join(sections)
        
        def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
            """評估預測 based on 類別 type."""
            data = self.get_split(split)
            if len(predictions) != len(data):
                return {
                    'accuracy': 0.0,
                    'error': f'Prediction count ({len(predictions)}) does not match data count ({len(data)})'
                }
            
            correct = 0
            total = len(predictions)
            category_stats = {}
            
            for i, (pred, example) in enumerate(zip(predictions, data)):
                expected = example['answer']
                category = example['category_id']
                choices_raw = example.get('choice', '') or ''
                
                # 初始化category stats
                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                
                is_correct = self._is_correct(pred, expected, category, choices_raw=choices_raw)
                if is_correct:
                    correct += 1
                    category_stats[category]['correct'] += 1
                category_stats[category]['total'] += 1
            
            # 建立results
            results = {
                'accuracy': correct / total if total > 0 else 0.0,
                'correct': correct,
                'total': total
            }
            
            # Add per-category 結果
            for category, stats in category_stats.items():
                cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                results[f'accuracy_{category}'] = cat_accuracy
                results[f'correct_{category}'] = stats['correct'] 
                results[f'total_{category}'] = stats['total']
            
            return results
        
        def _is_correct(self, prediction: str, expected: str, category: str,
                        choices_raw: str = '') -> bool:
            """檢查if 預測 is 正確 based on 類別.

            Always extracts only the 第一個 non-empty 行 of the 預測 so
            that 模型 continuations ("1\\nQuestion: ...\\n...") don't break matches.
            """
            # 擷取 第一個 meaningful 行
            first_line = ''
            for line in prediction.split('\n'):
                s = line.strip()
                if s:
                    first_line = s
                    break
            pred_clean = first_line.lower()
            expected_clean = str(expected).strip().lower()

            # Skip subjective 任務 marked with <LLMEval>
            if '<llmeval>' in expected_clean:
                return False

            # RG (A-E letter 答案): accept the letter directly, OR 檢查if the
            # 正確 choice 文字 appears in the 預測 (for 舊的 free-gen preds)
            if category.startswith('RG'):
                letters = 'abcde'
                # Direct letter match (第一個 char of 預測 is the 答案 letter)
                if pred_clean and pred_clean[0] == expected_clean:
                    return True
                # Soft match: 預測 contains the 文字 of the 正確 choice
                if choices_raw:
                    choices = [c.strip().lower() for c in choices_raw.split(';;')]
                    try:
                        idx = letters.index(expected_clean)
                        correct_text = choices[idx]
                        # 預測 contains 正確 choice 值
                        if correct_text and correct_text in pred_clean:
                            return True
                    except (ValueError, IndexError):
                        pass
                return False

            # RL1 (G/P 答案)
            if category == 'RL1':
                return pred_clean.startswith(expected_clean)

            # RL3/RL4, I1/I2 — numbered choice, 答案 is digit 字串
            if category.startswith('RL') or category.startswith('I'):
                # Accept 第一個 digit 字元 matching
                import re
                m = re.match(r'(\d+)', pred_clean)
                return bool(m and m.group(1) == expected_clean)

            # 詞彙: numbered choice, 答案 is digit 字串
            if category.startswith('V'):
                import re
                m = re.match(r'(\d+)', pred_clean)
                if m:
                    return m.group(1) == expected_clean
                # fallback: 第一個 char
                if pred_clean and pred_clean[0] == expected_clean:
                    return True
                return False

            # For 任務 with multiple 正確 答案 separated by ;;
            if ';;' in expected_clean:
                correct_answers = [ans.strip() for ans in expected_clean.split(';;')]
                return pred_clean in correct_answers

            # 預設: 完全匹配 of 第一個 行
            return pred_clean == expected_clean
        
        def get_ground_truth(self, split: str = "test") -> List[str]:
            """取得真值 for TextFRCT."""
            data = self.get_split(split)
            return [str(example['answer']) for example in data]


def create_textfrct_task(
    data_path: str = "dataset/TextFRCT.csv",
    skip_subjective: bool = False,
    categories: Optional[List[str]] = None,
    name: str = "textfrct"
) -> 'TextFRCTTask':
    """建立一個TextFRCT 任務 實例."""
    # Update 名稱 to reflect filtering
    if categories:
        name = f"textfrct_{'_'.join(categories)}"
    
    config = TaskConfig(
        name=name,
        description=f"TextFRCT evaluation dataset{' (filtered categories)' if categories else ''}",
        data_path=data_path,
        data_format="csv",
        input_column="question",
        output_column="answer",
        evaluation_metrics=["accuracy"],
        metadata={
            "skip_subjective": skip_subjective,
            "categories": categories
        }
    )
    
    return TextFRCTTask(config, skip_subjective=skip_subjective, categories=categories)
