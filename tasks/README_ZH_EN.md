# 任務評估系統 — 中英對照文檔 / Unified Task Evaluation System — Bilingual Documentation

---

## 功能特性 / Features

- **統一接口 / Unified Interface**：無論數據來源如何，所有任務使用相同 API / Common API for all tasks regardless of data source
- **多種數據格式 / Multiple Data Formats**：支持 CSV、JSON、JSONL 文件 / Support for CSV, JSON, and JSONL files
- **多種模型後端 / Multiple Model Backends**：vLLM、Transformers、OpenAI API、Together API
- **靈活的任務定義 / Flexible Task Definition**：通過配置文件輕鬆創建新任務 / Easy to create new tasks via configuration files
- **全面評估 / Comprehensive Evaluation**：按類別指標和詳細結果保存 / Per-category metrics and detailed result saving
- **可擴展設計 / Extensible Design**：易於添加新任務類型和評估指標 / Easy to add new task types and evaluation metrics

---

## 快速開始 / Quick Start

### 1. 基本用法 / Basic Usage

```python
from tasks import create_task_from_config, TaskEvaluator, ModelConfig, EvaluationConfig
from tasks.simple_icl_task import SimpleICLTask

# 創建任務 / Create a task
task = create_task_from_config('tasks/configs/simple_icl_tasks.json', SimpleICLTask)

# 配置模型 / Configure model
model_config = ModelConfig(
    model_id="allenai/OLMo-1B-hf",
    backend="vllm",
    temperature=0.0,
    max_tokens=10
)

# 配置評估 / Configure evaluation
eval_config = EvaluationConfig(output_dir="results")

# 運行評估 / Run evaluation
evaluator = TaskEvaluator(model_config, eval_config)
results = evaluator.evaluate_task(task)
```

### 2. 命令行用法 / Command Line Usage

```bash
# 使用 vLLM 評估簡單 ICL 任務 / Evaluate on Simple ICL tasks with vLLM
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id allenai/OLMo-1B-hf \
    --backend vllm \
    --output_dir results

# 使用 OpenAI API 評估 / Evaluate with OpenAI API
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id gpt-4o-mini-2024-07-18 \
    --backend openai \
    --api_key YOUR_API_KEY \
    --output_dir results
```

---

## 架構 / Architecture

### 核心組件 / Core Components

1. **BaseTask**：定義任務接口的抽象基類 / Abstract base class defining the task interface
2. **SimpleTask**：基礎實現，使用精確匹配評估 / Basic implementation with exact match evaluation
3. **SimpleICLTask**：基於類別演示的上下文學習專用任務 / Specialized task for in-context learning with category-based demonstrations
4. **TaskEvaluator**：支持多種模型後端的主評估引擎 / Main evaluation engine supporting multiple model backends
5. **配置類 / Configuration Classes**：任務、模型和評估的類型安全配置 / Type-safe configuration for tasks, models, and evaluation

### 任務配置 / Task Configuration

使用 JSON 配置文件定義任務 / Tasks are defined using JSON configuration files:

```json
{
  "name": "my_task",
  "description": "任務描述 / Description of the task",
  "data_path": "path/to/data.csv",
  "data_format": "csv",
  "input_column": "question",
  "output_column": "answer",
  "num_demonstrations": 5,
  "evaluation_metrics": ["accuracy"]
}
```

### 支持的後端 / Supported Backends

- **vLLM**：本地模型的高性能推理 / High-performance inference for local models
- **Transformers**：標準 HuggingFace transformers / Standard HuggingFace transformers
- **OpenAI**：OpenAI API 模型（GPT-4 等）/ OpenAI API models (GPT-4, etc.)
- **Together**：Together AI API 模型 / Together AI API models

---

## 創建新任務 / Creating New Tasks

### 1. 使用配置文件 / Using Configuration Files

創建 JSON 配置文件並使用 `SimpleTask` 類：

```python
from tasks.base_task import create_task_from_config, SimpleTask

task = create_task_from_config('my_task_config.json', SimpleTask)
```

### 2. 創建自定義任務類 / Creating Custom Task Classes

繼承 `BaseTask` 並重寫關鍵方法 / Inherit from `BaseTask` and override key methods:

```python
from tasks.base_task import BaseTask

class MyCustomTask(BaseTask):
    def build_prompt(self, instance):
        # 自定義提示構建邏輯 / Custom prompt building logic
        return f"Question: {instance['input']}\nAnswer:"

    def evaluate(self, predictions, split="test", **kwargs):
        # 自定義評估邏輯 / Custom evaluation logic
        ground_truth = self.get_ground_truth(split)
        accuracy = compute_accuracy(predictions, ground_truth)
        return {"accuracy": accuracy}
```

---

## 示例 / Examples

### 簡單 ICL 任務 / Simple ICL Task

`SimpleICLTask` 演示了基於類別的上下文學習：

```python
# 每個類別獲取特定演示 / Each category gets specific demonstrations
category_demonstrations = {
    "uppercase": ["a -> A", "c -> C"],
    "lowercase": ["A -> a", "C -> c"],
    "translate_eng_fr": ["hello -> bonjour", "goodbye -> au revoir"]
}

# 使用類別特定示例構建提示 / Prompts are built with category-specific examples
# 大寫示例 / For uppercase: "a -> A\nc -> C\nb ->"
```

### 評估結果 / Evaluation Results

系統提供全面的評估指標 / The system provides comprehensive evaluation metrics:

```python
{
    "accuracy": 0.8966,           # 總體準確率 / Overall accuracy
    "correct": 104,               # 正確預測總數 / Total correct predictions
    "total": 116,                 # 示例總數 / Total examples
    "accuracy_uppercase": 0.8750, # 按類別準確率 / Per-category accuracy
    "accuracy_lowercase": 0.8750,
    # ... 更多按類別指標 / more per-category metrics
}
```

---

## 文件結構 / File Structure

```
tasks/
├── __init__.py              # 任務注冊表和導入 / Task registry and imports
├── base_task.py             # 基礎類和接口 / Base classes and interfaces
├── evaluator.py             # 主評估引擎 / Main evaluation engine
├── simple_icl_task.py       # ICL 任務實現 / ICL task implementation
├── textfrct_task.py         # TextFRCT 集成（可選）/ TextFRCT integration (optional)
└── configs/
    └── simple_icl_tasks.json # 任務配置示例 / Task configuration example
```

---

## 從舊代碼遷移 / Migration from Legacy Code

統一系統替換了以前分散的方法 / The unified system replaces the previous scattered approaches:

### 之前（多種方法）/ Before (Multiple Approaches)
- `run_interp.ipynb`：帶硬編碼示例的手動 ICL / Manual ICL with hardcoded examples
- `run.ipynb`：使用 OpenAI API 的 TextFRCT / TextFRCT with OpenAI API
- `models/evaluate_models.py`：帶任務注冊表的 vLLM / vLLM with task registry
- `scripts/evaluate_textfrct.py`：使用多個後端的 TextFRCT / TextFRCT with multiple backends

### 之後（統一系統）/ After (Unified System)
- 單一 `TaskEvaluator` 類 / Single `TaskEvaluator` class
- 通用任務接口 / Common task interface
- 統一配置格式 / Unified configuration format
- 一致的結果格式 / Consistent result format
- 支持所有以前的後端 / Support for all previous backends

### 遷移步驟 / Migration Steps

1. **替換 Notebook 評估 / Replace notebook evaluations**：
   ```python
   # 舊版：手動構建提示和評估 / Old: Manual prompt building and evaluation
   # 新版：使用 SimpleICLTask 和 TaskEvaluator / New: Use SimpleICLTask and TaskEvaluator
   ```

2. **替換基於腳本的評估 / Replace script-based evaluations**：
   ```bash
   # 舊版：多個不同腳本 / Old: Multiple different scripts
   # 新版：單一 run_unified_eval.py，使用不同 --task_type / New: Single run_unified_eval.py with different --task_type
   ```

3. **更新模型配置 / Update model configurations**：
   ```python
   # 舊版：各種參數格式 / Old: Various parameter formats
   # 新版：帶驗證的 ModelConfig 數據類 / New: ModelConfig dataclass with validation
   ```

---

## 測試 / Testing

運行測試套件以驗證系統 / Run the test suite to verify the system:

```bash
python test_unified_system.py
```

這將 / This will:
- 顯示不同類別的示例提示 / Show example prompts for different categories
- 使用模擬預測運行模擬評估 / Run mock evaluation with simulated predictions
- 顯示按類別的準確率指標 / Display per-category accuracy metrics

---

## 未來擴展 / Future Extensions

系統設計為易於擴展 / The system is designed to be easily extensible:

1. **新任務類型 / New Task Types**：為特定評估需求添加新任務類 / Add new task classes for specific evaluation needs
2. **新後端 / New Backends**：添加對其他模型服務框架的支持 / Add support for additional model serving frameworks
3. **新指標 / New Metrics**：使用領域特定指標擴展評估 / Extend evaluation with domain-specific metrics
4. **批量處理 / Batch Processing**：添加對評估多個模型/任務的支持 / Add support for evaluating multiple models/tasks
5. **緩存 / Caching**：為昂貴的評估添加結果緩存 / Add result caching for expensive evaluations

---

> 📄 本文件為中英對照版本。英文原版請參閱 [README.md](README.md)。
> This is the bilingual version. For the original English documentation, see [README.md](README.md).
