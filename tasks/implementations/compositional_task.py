"""組合式 任務 that chains multiple atomic 操作."""

import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

from tasks.base_task import BaseTask, TaskConfig


# =============================================================================
# Atomic 操作 Registry
# =============================================================================

# 純字串 操作 (no external 資料 needed)
STRING_OPERATIONS: Dict[str, Callable[[str], str]] = {
    "uppercase": lambda x: x.upper(),
    "lowercase": lambda x: x.lower(),
    "reverse": lambda x: x[::-1],
    "first_letter": lambda x: x[0] if x else "",
    "last_letter": lambda x: x[-1] if x else "",
}


def load_lookup_tables() -> Dict[str, Dict[str, str]]:
    """載入基於查找的 操作 from simple.csv.
    
    回傳a 字典 mapping 操作 名稱 to their 查找 tables.
    """
    csv_path = Path(__file__).parent.parent.parent / "dataset" / "simple.csv"
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    
    # 類別 that can be used as 查找 操作
    lookup_categories = [
        "translate_eng_fr", "translate_fr_eng",
        "translate_eng_sp", "translate_sp_eng",
        "present_to_gerund", "singular_to_plural",
    ]
    
    lookup_tables = {}
    for category in lookup_categories:
        cat_data = df[df["category_name"] == category]
        if not cat_data.empty:
            lookup_tables[category] = dict(zip(cat_data["question"], cat_data["answer"]))

    # Backfill eng->sp from sp->eng when simple.csv lacks translate_eng_sp rows.
    # Keep the 第一個 observed Spanish form for each English token.
    if "translate_eng_sp" not in lookup_tables and "translate_sp_eng" in lookup_tables:
        sp_to_eng = lookup_tables["translate_sp_eng"]
        eng_to_sp: Dict[str, str] = {}
        for sp, eng in sp_to_eng.items():
            eng_norm = str(eng).strip().lower()
            if eng_norm and eng_norm not in eng_to_sp:
                eng_to_sp[eng_norm] = str(sp)
        if eng_to_sp:
            lookup_tables["translate_eng_sp"] = eng_to_sp
    
    return lookup_tables


# Global 查找 tables (已載入 once)
LOOKUP_TABLES: Dict[str, Dict[str, str]] = {}


def get_lookup_operation(op_name: str) -> Optional[Callable[[str], str]]:
    """取得a 基於查找的 操作 function."""
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    if op_name in LOOKUP_TABLES:
        table = LOOKUP_TABLES[op_name]
        return lambda x: table.get(x, x)  # 回傳original if not found
    return None


def get_operation(op_name: str) -> Callable[[str], str]:
    """取得an 操作 function by 名稱 (字串 or 基於查找的)."""
    if op_name in STRING_OPERATIONS:
        return STRING_OPERATIONS[op_name]
    
    lookup_op = get_lookup_operation(op_name)
    if lookup_op:
        return lookup_op
    
    raise ValueError(f"Unknown operation: {op_name}")


def apply_composition(input_str: str, operations: List[str]) -> str:
    """Apply a sequence of 操作 to an 輸入 字串."""
    result = input_str
    for op_name in operations:
        op_func = get_operation(op_name)
        result = op_func(result)
    return result


def parse_operations(operations_str: str) -> List[str]:
    """Parse 操作 字串 like 'uppercase+reverse' into 列表."""
    return operations_str.split("+")


# =============================================================================
# Predefined 組合
# =============================================================================

# 純字串 組合 (any 輸入 works)
STRING_COMPOSITIONS = {
    # 2-操作: case + manipulation
    "upper_reverse": ["uppercase", "reverse"],
    "lower_reverse": ["lowercase", "reverse"],
    "upper_first": ["uppercase", "first_letter"],
    "lower_first": ["lowercase", "first_letter"],
    "upper_last": ["uppercase", "last_letter"],
    "lower_last": ["lowercase", "last_letter"],
    "reverse_first": ["reverse", "first_letter"],
    "reverse_last": ["reverse", "last_letter"],
    
    # 3-操作 chains

}

# 基於查找的 組合 (require 特定 輸入 domains)
# 格式: (composition_name, 操作, source_lookup_table)
# The source_lookup_table determines 有效 inputs
LOOKUP_COMPOSITIONS = {
    # Translation eng->fr + 字串 ops (complete coverage)
    "translate_eng_fr_upper": (["translate_eng_fr", "uppercase"], "translate_eng_fr"),
    "translate_eng_fr_lower": (["translate_eng_fr", "lowercase"], "translate_eng_fr"),
    "translate_eng_fr_reverse": (["translate_eng_fr", "reverse"], "translate_eng_fr"),
    "translate_eng_fr_first": (["translate_eng_fr", "first_letter"], "translate_eng_fr"),
    "translate_eng_fr_last": (["translate_eng_fr", "last_letter"], "translate_eng_fr"),
    
    # Translation eng->sp + 字串 ops (complete coverage)
    "translate_eng_sp_upper": (["translate_eng_sp", "uppercase"], "translate_eng_sp"),
    "translate_eng_sp_lower": (["translate_eng_sp", "lowercase"], "translate_eng_sp"),
    "translate_eng_sp_reverse": (["translate_eng_sp", "reverse"], "translate_eng_sp"),
    "translate_eng_sp_first": (["translate_eng_sp", "first_letter"], "translate_eng_sp"),
    "translate_eng_sp_last": (["translate_eng_sp", "last_letter"], "translate_eng_sp"),
    
    # Translation fr->eng + 字串 ops (complete coverage)
    "translate_fr_eng_upper": (["translate_fr_eng", "uppercase"], "translate_fr_eng"),
    "translate_fr_eng_lower": (["translate_fr_eng", "lowercase"], "translate_fr_eng"),
    "translate_fr_eng_reverse": (["translate_fr_eng", "reverse"], "translate_fr_eng"),
    "translate_fr_eng_first": (["translate_fr_eng", "first_letter"], "translate_fr_eng"),
    "translate_fr_eng_last": (["translate_fr_eng", "last_letter"], "translate_fr_eng"),
    
    # Translation sp->eng + 字串 ops (complete coverage)
    "translate_sp_eng_upper": (["translate_sp_eng", "uppercase"], "translate_sp_eng"),
    "translate_sp_eng_lower": (["translate_sp_eng", "lowercase"], "translate_sp_eng"),
    "translate_sp_eng_reverse": (["translate_sp_eng", "reverse"], "translate_sp_eng"),
    "translate_sp_eng_first": (["translate_sp_eng", "first_letter"], "translate_sp_eng"),
    "translate_sp_eng_last": (["translate_sp_eng", "last_letter"], "translate_sp_eng"),
    
    # Morphological + 字串 ops (complete coverage)
    "gerund_upper": (["present_to_gerund", "uppercase"], "present_to_gerund"),
    "gerund_lower": (["present_to_gerund", "lowercase"], "present_to_gerund"),
    "gerund_reverse": (["present_to_gerund", "reverse"], "present_to_gerund"),
    "gerund_first": (["present_to_gerund", "first_letter"], "present_to_gerund"),
    "plural_upper": (["singular_to_plural", "uppercase"], "singular_to_plural"),
    "plural_lower": (["singular_to_plural", "lowercase"], "singular_to_plural"),
    "plural_reverse": (["singular_to_plural", "reverse"], "singular_to_plural"),
    "plural_first": (["singular_to_plural", "first_letter"], "singular_to_plural"),
    
    # 3-操作 chains with 查找
    "gerund_upper_reverse": (["present_to_gerund", "uppercase", "reverse"], "present_to_gerund"),
    "plural_upper_reverse": (["singular_to_plural", "uppercase", "reverse"], "singular_to_plural"),
    "translate_eng_fr_upper_reverse": (["translate_eng_fr", "uppercase", "reverse"], "translate_eng_fr"),
    "translate_eng_sp_upper_reverse": (["translate_eng_sp", "uppercase", "reverse"], "translate_eng_sp"),
}


# =============================================================================
# 輸入 Pools
# =============================================================================


def load_atomic_operation_inputs() -> Dict[str, List[str]]:
    """載入actual 輸入 範例 for each atomic 操作 from simple.csv.
    
    This ensures 組合式 任務 use in-distribution inputs from component 任務.
    
    回傳：
        字典 mapping 操作 名稱 to their 輸入 pools from simple.csv
    """
    csv_path = Path(__file__).parent.parent.parent / "dataset" / "simple.csv"
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Map simple.csv 類別 名稱 to 操作 名稱
    category_to_op = {
        "uppercase": "uppercase",
        "lowercase": "lowercase",
        "first_letter": "first_letter",
        "last_letter": "last_letter",
        "reverse": "reverse",  # May not exist in simple.csv
    }
    
    operation_inputs = {}
    for category, op_name in category_to_op.items():
        cat_data = df[df["category_name"] == category]
        if not cat_data.empty:
            operation_inputs[op_name] = cat_data["question"].unique().tolist()
    
    return operation_inputs


def load_generic_string_inputs() -> List[str]:
    """載入a broad in-domain 詞 pool for 純字串 組合.

    This stays grounded in simple.csv rather than introducing 合成 字串.
    Restrict to 單 token alphabetic 字串 longer than one 字元 so
    reverse/第一個/最後一個 操作 remain meaningful.
    """
    csv_path = Path(__file__).parent.parent.parent / "dataset" / "simple.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    questions = df["question"].dropna().astype(str)
    values = {
        value
        for value in questions
        if " " not in value and len(value) > 1 and all(ch.isalpha() for ch in value)
    }
    return sorted(values)


def get_seed_string_inputs(operations: List[str], operation_inputs: Dict[str, List[str]]) -> List[str]:
    """Choose a meaningful 輸入 pool for 純字串 組合.

    Using the 第一個 操作's atomic 資料集 directly makes chains like
    lowercase+reverse collapse into one-character 範例 because the lowercase
    任務 only contains A-Z. Use a richer in-domain 詞 pool instead, and bias
    its casing so case-conversion 操作 still do real work.
    """
    generic_inputs = load_generic_string_inputs()
    first_op = operations[0]

    if not generic_inputs:
        return operation_inputs.get(first_op, [])

    if "lowercase" in operations and "uppercase" not in operations:
        return [value.upper() for value in generic_inputs]

    if "uppercase" in operations and "lowercase" not in operations:
        return [value.lower() for value in generic_inputs]

    return generic_inputs


def get_string_composition_inputs(operations: List[str], strict_chain: bool = False) -> List[str]:
    """取得input pool for a 字串組合.
    
    參數：
        操作: 列表 of 操作 名稱 in the 組合
        strict_chain: If True, only use inputs where intermediate 結果 are also 
                     in-distribution (Approach B). If False, use a broad in-domain 字串 pool.
    
    回傳：
        列表 of 有效 輸入 字串 for this 組合
    """
    operation_inputs = load_atomic_operation_inputs()
    
    first_op = operations[0]
    inputs = get_seed_string_inputs(operations, operation_inputs)
    if not inputs:
        return []
    
    # Approach A (預設): Use the richer 種子 inputs.
    if not strict_chain:
        return inputs
    
    # Approach B (strict): Only keep inputs where entire chain is in-distribution
    if len(operations) < 2:
        return inputs
    
    second_op = operations[1]
    
    # If the 下一個 op has no atomic pool, keep the richer 種子 inputs.
    if second_op not in operation_inputs:
        return inputs
    
    valid_for_op2 = set(operation_inputs[second_op])
    
    # 篩選 to inputs where intermediate 結果 is in-distribution for op2
    valid_inputs = []
    for inp in inputs:
        try:
            op_func = get_operation(first_op)
            intermediate = op_func(inp)
            if intermediate in valid_for_op2:
                valid_inputs.append(inp)
        except Exception:
            continue
    
    return valid_inputs  # May 回傳empty 列表 if no inputs satisfy strict chain requirement


# 操作 that benefit from 字元間距
SPACING_BENEFITS = {"reverse", "first_letter", "last_letter"}


def add_spaces(s: str) -> str:
    """Add spaces between each 字元."""
    return " ".join(list(s))


def remove_spaces(s: str) -> str:
    """Remove spaces from a 字串."""
    return s.replace(" ", "")


def composition_benefits_from_spacing(operations: List[str]) -> bool:
    """檢查if a 組合 would benefit from 字元間距."""
    return any(op in SPACING_BENEFITS for op in operations)


def get_lookup_inputs(lookup_name: str) -> List[str]:
    """取得valid inputs for a 基於查找的 操作."""
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    if lookup_name in LOOKUP_TABLES:
        return list(LOOKUP_TABLES[lookup_name].keys())
    return []


def build_lookup_example(input_str: str, operations: List[str]) -> tuple[str, str]:
    """建立a 基於查找的 範例 輸入/輸出 pair for a 組合.

    For 查找 chains ending with lowercase (and no uppercase op), expose an
    uppercase 輸入 so the lowercase step is not a no-op while preserving 查找
    validity via the original key.
    """
    display_input = input_str

    if "lowercase" in operations and "uppercase" not in operations:
        display_input = input_str.upper()

        # Keep 查找 domain 有效 with the original key, then force lowercase
        # to do real work by uppercasing the intermediate 字串 第一個.
        result = get_operation(operations[0])(input_str)
        result = result.upper()
        for op_name in operations[1:]:
            result = get_operation(op_name)(result)
        return display_input, result

    return display_input, apply_composition(input_str, operations)


# =============================================================================
# 組合式 任務 實作
# =============================================================================

class CompositionalTask(BaseTask):
    """A 任務 that chains multiple atomic 操作.
    
    This 任務 支援:
    - 載入 from CSV 檔案 (dataset/compositional.csv)
    - Auto-generating 資料 if CSV doesn't exist
    - Subtask filtering via category_name (e.g., 組合式:upper_reverse)
    - Spaced mode for character-level 操作 (spaced=True)
    
    範例：
        # 載入all 組合式 任務
        任務 = get_task("compositional")
        
        # 載入specific 組合
        任務 = get_task("compositional:upper_reverse")
        
        # 載入with spacing for character-level 操作
        任務 = get_task("compositional:upper_reverse", spaced=True)
    """
    
    TASK_NAME = "compositional"  # 自動註冊名稱
    
    def __init__(self, config: TaskConfig, spaced: bool = False):
        """初始化compositional 任務.
        
        參數：
            設定: 任務 設定
            spaced: If True, add spaces between 字元 in 輸入/輸出
        """
        self.spaced = spaced
        super().__init__(config)
    
    def _load_data(self):
        """載入compositional 任務 資料 from CSV or 生成it."""
        # Use spaced CSV if in spaced mode
        if self.spaced:
            data_path = Path(__file__).parent.parent.parent / "dataset" / "compositional_spaced.csv"
        else:
            data_path = Path(__file__).parent.parent.parent / "dataset" / "compositional.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            df = df.fillna("")
            self.data = df.to_dict("records")
        else:
            # 生成data if CSV doesn't exist
            self._generate_data()
            # Optionally 儲存to CSV for future runs
            self._save_data(data_path)
    
    def _generate_data(self, strict_chain: bool = False):
        """生成compositional 範例 programmatically.
        
        參數：
            strict_chain: If True, use Approach B (only inputs where entire chain is in-distribution).
                         If False, use Approach A (use 第一個 操作's inputs).
        """
        examples = []
        
        # 生成examples for 純字串 組合
        for comp_name, ops in STRING_COMPOSITIONS.items():
            # 取得appropriate 輸入 pool based on approach
            valid_inputs = get_string_composition_inputs(ops, strict_chain=strict_chain)
            
            for input_str in valid_inputs:
                try:
                    output = apply_composition(input_str, ops)
                    
                    if self.spaced:
                        examples.append({
                            "input": add_spaces(input_str),
                            "output": add_spaces(output),
                            "original_input": input_str,
                            "original_output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                            "spaced": True,
                        })
                    else:
                        examples.append({
                            "input": input_str,
                            "output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                        })
                except Exception as e:
                    print(f"Warning: Failed to generate {comp_name} for '{input_str}': {e}")
                    continue
        
        # 生成examples for 基於查找的 組合
        for comp_name, (ops, source_lookup) in LOOKUP_COMPOSITIONS.items():
            valid_inputs = get_lookup_inputs(source_lookup)
            for input_str in valid_inputs:
                try:
                    display_input, output = build_lookup_example(input_str, ops)
                    # Skip if 查找 returned original (meaning 查找 failed)
                    if ops[0] in LOOKUP_TABLES and output == input_str:
                        continue
                    
                    if self.spaced:
                        examples.append({
                            "input": add_spaces(display_input),
                            "output": add_spaces(output),
                            "original_input": display_input,
                            "original_output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                            "spaced": True,
                        })
                    else:
                        examples.append({
                            "input": display_input,
                            "output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                        })
                except Exception as e:
                    print(f"Warning: Failed to generate {comp_name} for '{input_str}': {e}")
                    continue
        
        self.data = examples
    
    def _save_data(self, path: Path):
        """儲存generated 資料 to CSV 以確保可重現性."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)
        print(f"Saved compositional data to {path}")
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """建立ICL 提示 with 示範 from 相同類別.
        
        參數：
            實例: Test 實例 with 'input', 'output', 'category_name'
            num_shots: 數字 of 示範 範例
            
        回傳：
            格式化 提示 字串
        """
        category = instance.get("category_name", "")
        
        # 取得demos from 相同類別, excluding test 實例
        demos = [
            ex for ex in self.data 
            if ex.get("category_name") == category and ex["input"] != instance["input"]
        ]
        random.shuffle(demos)
        demos = demos[:num_shots]
        
        # 建立提示 with 簡單 arrow 格式化(like simple_icl)
        prompt_parts = []
        
        # Add 示範
        for demo in demos:
            prompt_parts.append(f"{demo['input']} -> {demo['output']}")
        
        # Add test 實例 (without 答案)
        prompt_parts.append(f"{instance['input']} ->")
        
        return "\n".join(prompt_parts)
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """評估預測 against 真值.
        
        參數：
            預測: 列表 of 模型 預測
            切分: 資料 切分 to 評估on
            
        回傳：
            字典 with 評估 metrics
        """
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # 預處理 預測
        processed_predictions = []
        for pred in predictions:
            # Clean 預測: take 第一個 行, strip whitespace
            pred_clean = pred.strip().split("\n")[0].strip()
            
            # Remove leading arrow if present
            if pred_clean.startswith("->"):
                pred_clean = pred_clean[2:].strip()
            
            # For spaced mode, keep spaces; otherwise take 第一個 詞
            if self.spaced:
                # Keep the full spaced 輸出
                processed_predictions.append(pred_clean)
            else:
                # Take 第一個 詞 only
                pred_clean = pred_clean.split()[0] if pred_clean.split() else pred_clean
                processed_predictions.append(pred_clean)
        
        def matches(pred: str, gt: str) -> bool:
            """檢查if 預測 matches 真值, handling spaced mode."""
            pred_norm = pred.lower().strip()
            gt_norm = gt.lower().strip()
            
            if pred_norm == gt_norm:
                return True
            
            # For spaced mode, also try comparing without spaces
            if self.spaced:
                pred_unspaced = remove_spaces(pred_norm)
                gt_unspaced = remove_spaces(gt_norm)
                return pred_unspaced == gt_unspaced
            
            return False
        
        # Overall 準確率
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if matches(pred, gt))
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
            if matches(pred, gt):
                category_stats[category]["correct"] += 1
        
        # Calculate per-category 準確率
        for category, stats in category_stats.items():
            results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
            results[f"correct_{category}"] = stats["correct"]
            results[f"total_{category}"] = stats["total"]
        
        return results
    
    @property
    def task_type(self) -> str:
        return "compositional_spaced" if self.spaced else "compositional"
    
    @property
    def description(self) -> str:
        return "Compositional task that chains multiple atomic string operations"


# =============================================================================
# 工具 Functions
# =============================================================================

def generate_compositional_csv(output_path: str = None, spaced: bool = False, strict_chain: bool = False):
    """生成the compositional.csv or compositional_spaced.csv 檔案.
    
    參數：
        output_path: 路徑 to 儲存CSV. 預設s to dataset/compositional.csv or compositional_spaced.csv
        spaced: If True, 生成spaced version with spaces between 字元
        strict_chain: If True, use Approach B (strict in-distribution chains). 
                     If False, use Approach A (第一個 op's inputs).
    """
    if output_path is None:
        if spaced:
            output_path = Path(__file__).parent.parent.parent / "dataset" / "compositional_spaced.csv"
        else:
            output_path = Path(__file__).parent.parent.parent / "dataset" / "compositional.csv"
    else:
        output_path = Path(output_path)
    
    # Ensure 查找 tables are 已載入
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    examples = []
    
    # 生成純字串 組合 using Approach A or B
    for comp_name, ops in STRING_COMPOSITIONS.items():
        valid_inputs = get_string_composition_inputs(ops, strict_chain=strict_chain)
        
        for input_str in valid_inputs:
            try:
                output = apply_composition(input_str, ops)
                
                if spaced:
                    examples.append({
                        "input": add_spaces(input_str),
                        "output": add_spaces(output),
                        "original_input": input_str,
                        "original_output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                        "spaced": True,
                    })
                else:
                    examples.append({
                        "input": input_str,
                        "output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                    })
            except Exception:
                continue
    
    # 生成基於查找的 組合
    for comp_name, (ops, source_lookup) in LOOKUP_COMPOSITIONS.items():
        valid_inputs = get_lookup_inputs(source_lookup)
        for input_str in valid_inputs:
            try:
                display_input, output = build_lookup_example(input_str, ops)
                
                if spaced:
                    examples.append({
                        "input": add_spaces(display_input),
                        "output": add_spaces(output),
                        "original_input": display_input,
                        "original_output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                        "spaced": True,
                    })
                else:
                    examples.append({
                        "input": display_input,
                        "output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                    })
            except Exception:
                continue
    
    df = pd.DataFrame(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # 摘要
    string_comps = len(STRING_COMPOSITIONS)
    lookup_comps = len(LOOKUP_COMPOSITIONS)
    total_comps = string_comps + lookup_comps
    
    mode = "spaced" if spaced else "normal"
    approach = "Approach B (strict chains)" if strict_chain else "Approach A (first op's inputs)"
    print(f"Generated {len(examples)} {mode} examples across {total_comps} compositions using {approach}")
    print(f"  - {string_comps} string compositions (pure string ops)")
    print(f"  - {lookup_comps} lookup compositions (translation/morphological + string ops)")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    import sys
    # 生成both normal and spaced CSV 檔案
    if len(sys.argv) > 1 and sys.argv[1] == "--spaced":
        generate_compositional_csv(spaced=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        generate_compositional_csv(spaced=False)
        generate_compositional_csv(spaced=True)
    else:
        generate_compositional_csv(spaced=False)
