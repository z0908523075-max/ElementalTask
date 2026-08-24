# 統一任務評估系統

此目錄包含一個統一的任務評估框架，整合了專案先前使用的各種方法。此系統為不同任務上的模型評估提供共同介面，同時支援多種資料格式與模型後端。

## 功能

- **統一介面**：無論資料來源為何，所有任務都共用 API
- **多種資料格式**：支援 CSV、JSON 與 JSONL 檔案
- **多種模型後端**：vLLM、Transformers、OpenAI API、Together API
- **彈性的任務定義**：可透過設定檔輕鬆建立新任務
- **完整評估**：提供各類別指標與詳細結果儲存
- **可擴充設計**：易於新增任務類型與評估指標

## 快速開始

### 1. 基本用法

```python
from tasks import create_task_from_config, TaskEvaluator, ModelConfig, EvaluationConfig
from tasks.simple_icl_task import SimpleICLTask

# 建立任務
task = create_task_from_config('tasks/configs/simple_icl_tasks.json', SimpleICLTask)

# 設定模型
model_config = ModelConfig(
    model_id="allenai/OLMo-1B-hf",
    backend="vllm",
    temperature=0.0,
    max_tokens=10
)

# 設定評估
eval_config = EvaluationConfig(output_dir="results")

# 執行評估
evaluator = TaskEvaluator(model_config, eval_config)
results = evaluator.evaluate_task(task)
```

### 2. 命令列用法

```bash
# 使用 vLLM 評估 Simple ICL 任務
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id allenai/OLMo-1B-hf \
    --backend vllm \
    --output_dir results

# 使用 OpenAI API 評估
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id gpt-4o-mini-2024-07-18 \
    --backend openai \
    --api_key YOUR_API_KEY \
    --output_dir results
```

## 架構

### 核心元件

1. **BaseTask**：定義任務介面的抽象基底類別
2. **SimpleTask**：具有 exact match 評估的基本實作
3. **SimpleICLTask**：針對具類別示範的 in-context learning 所設計的專用任務
4. **TaskEvaluator**：支援多種模型後端的主要評估引擎
5. **Configuration Classes**：用於任務、模型與評估的 type-safe 設定

### 任務設定

任務使用 JSON 設定檔來定義：

```json
{
  "name": "my_task",
  "description": "Description of the task",
  "data_path": "path/to/data.csv",
  "data_format": "csv",
  "input_column": "question",
  "output_column": "answer",
  "num_demonstrations": 5,
  "evaluation_metrics": ["accuracy"]
}
```

### 支援的後端

- **vLLM**：用於本機模型的高效能推論
- **Transformers**：標準 HuggingFace transformers
- **OpenAI**：OpenAI API 模型（GPT-4 等）
- **Together**：Together AI API 模型

## 建立新任務

### 1. 使用設定檔

建立 JSON 設定檔，並使用 `SimpleTask` 類別：

```python
from tasks.base_task import create_task_from_config, SimpleTask

task = create_task_from_config('my_task_config.json', SimpleTask)
```

### 2. 建立自訂任務類別

繼承 `BaseTask` 並覆寫關鍵方法：

```python
from tasks.base_task import BaseTask

class MyCustomTask(BaseTask):
    def build_prompt(self, instance):
        # 自訂 prompt 建構邏輯
        return f"Question: {instance['input']}\nAnswer:"
    
    def evaluate(self, predictions, split="test", **kwargs):
        # 自訂評估邏輯
        ground_truth = self.get_ground_truth(split)
        accuracy = compute_accuracy(predictions, ground_truth)
        return {"accuracy": accuracy}
```

## 範例

### Simple ICL Task

`SimpleICLTask` 展示了基於類別的 in-context learning：

```python
# 每個類別都會取得特定示範
category_demonstrations = {
    "uppercase": ["a -> A", "c -> C"],
    "lowercase": ["A -> a", "C -> c"],
    "translate_eng_fr": ["hello -> bonjour", "goodbye -> au revoir"]
}

# Prompts 會用類別特定的範例來建構
# 對 uppercase 來說："a -> A\nc -> C\nb ->"
```

### 評估結果

此系統提供完整的評估指標：

```python
{
    "accuracy": 0.8966,           # 整體準確率
    "correct": 104,               # 總正確預測數
    "total": 116,                 # 總範例數
    "accuracy_uppercase": 0.8750, # 各類別準確率
    "accuracy_lowercase": 0.8750,
    # ... 更多各類別指標
}
```

## 檔案結構

```
tasks/
├── __init__.py              # 任務 registry 與 imports
├── base_task.py             # 基底類別與介面
├── evaluator.py             # 主要評估引擎
├── simple_icl_task.py       # ICL 任務實作
├── textfrct_task.py         # TextFRCT 整合（可選）
└── configs/
    └── simple_icl_tasks.json # 任務設定範例
```

## 從舊版程式碼遷移

統一系統取代了先前分散的做法：

### 之前（多種做法）
- `run_interp.ipynb`：使用硬編碼範例的手動 ICL
- `run.ipynb`：使用 OpenAI API 的 TextFRCT
- `models/evaluate_models.py`：搭配 task registry 的 vLLM
- `scripts/evaluate_textfrct.py`：支援多種後端的 TextFRCT

### 之後（統一系統）
- 單一 `TaskEvaluator` 類別
- 共通任務介面
- 統一設定格式
- 一致的結果格式
- 支援所有先前的後端

### 遷移步驟

1. **取代 notebook 評估**：
   ```python
   # 舊：手動建構 prompt 與評估
   # 新：使用 SimpleICLTask 與 TaskEvaluator
   ```

2. **取代腳本式評估**：
   ```bash
   # 舊：多個不同腳本
   # 新：單一 run_unified_eval.py，搭配不同的 --task_type
   ```

3. **更新模型設定**：
   ```python
   # 舊：各種不同的參數格式
   # 新：具驗證功能的 ModelConfig dataclass
   ```

## 測試

執行測試套件以驗證系統：

```bash
python test_unified_system.py
```

這會：
- 顯示不同類別的範例 prompts
- 使用模擬預測執行 mock evaluation
- 顯示各類別準確率指標

## 未來擴充

此系統設計為可輕鬆擴充：

1. **新任務類型**：針對特定評估需求新增任務類別
2. **新後端**：新增對其他模型服務框架的支援
3. **新指標**：擴充為領域特定的評估指標
4. **批次處理**：新增同時評估多個模型／任務的支援
5. **快取**：為昂貴的評估加入結果快取
