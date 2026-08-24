# 函數向量提取 — 中英對照文檔 / Function Vectors Extraction — Bilingual Documentation

本目錄包含從語言模型提取**函數向量**的工具。函數向量是 L2 歸一化向量，捕捉注意力頭如何在上下文學習（ICL）示例和對照示例（輸入→輸出映射被打亂的示例）之間，對殘差流產生不同貢獻。它們代表了模型學習到的任務特定信息。

This directory contains tools for extracting **function vectors** from language models. Function vectors are L2-normalized vectors that capture how attention heads contribute to the residual stream differently for in-context learning (ICL) examples vs. control examples (where input→output mappings are shuffled). They represent task-specific information learned by the model.

---

## 目錄 / Table of Contents

- [快速開始 / Quick Start](#快速開始--quick-start)
- [兩種提取接口 / Two Extraction Interfaces](#兩種提取接口--two-extraction-interfaces)
  - [簡單接口 / Simple Interface](#1-簡單接口（推薦）--simple-interface-recommended)
  - [高級接口 / Advanced Interface](#2-高級接口--advanced-interface-full-control)
- [加載模型檢查點 / Loading Model Checkpoints](#加載模型檢查點--loading-model-checkpoints)
- [構建技能基底 / Building a Skill Basis](#構建多任務技能基底--building-a-skill-basis-from-multiple-tasks)
- [發現可用任務 / Discovering Available Tasks](#發現可用任務--discovering-available-tasks)
- [配置參數 / Configuration Parameters](#關鍵配置參數--key-configuration-parameters)
- [工作原理 / How It Works](#工作原理--how-it-works)
- [運行測試 / Running Tests](#運行測試--running-tests)

---

## 快速開始 / Quick Start

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 一行提取函數向量 / Extract function vector with one line
function_vec = extract_function_vector_simple("basic_arithmetic", num_samples=10)

print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")
print(f"L2 norm: {function_vec.function_vec.dot(function_vec.function_vec):.6f}")  # ~1.0
```

---

## 兩種提取接口 / Two Extraction Interfaces

### 1. 簡單接口（推薦）/ Simple Interface (Recommended)

簡單接口提供具有自動配置的一站式函數提取。適用於快速原型設計、標準用例和入門。

The simple interface provides one-stop function extraction with automatic configuration. Best for rapid prototyping, standard use cases, and getting started.

#### 基本用法 / Basic Usage

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 最小用法 - 使用所有默認值 / Minimal usage - uses all defaults
function_vec = extract_function_vector_simple("basic_arithmetic")

# 自定義參數 / With custom parameters
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="gpt2",          # 或 / or "distilgpt2", "EleutherAI/gpt-j-6B" 等
    num_samples=20,              # 使用的示例數量 / Number of examples to use
    device="cuda",               # "auto", "cuda" 或 / or "cpu"
    layer_idx=11                 # 特定層（None = 使用最後一層）/ Specific layer (None = use last layer)
)

print(f"Task: {function_vec.task_name}")
print(f"Shape: {function_vec.function_vec.shape}")
print(f"Normalization: {function_vec.normalization}")  # "l2"
```

#### 可用任務 / Available Tasks

任務從任務注冊表中自動發現。可用任務包括 / Tasks are auto-discovered from the task registry. Available tasks include:
- `basic_arithmetic` — 基礎算術操作 / Basic arithmetic operations
- `simple_icl` — 簡單上下文學習任務 / Simple in-context learning tasks
- `simple` — 簡單任務示例 / Simple task examples
- `textfrct` — TextFRCT 數據集任務 / TextFRCT dataset tasks
- `part_of_speech` — 詞性識別 / Part of speech identification
- `token_reversal` — 令牌反轉操作 / Token reversal operations
- `math` — 數學推理任務 / Mathematical reasoning tasks
- `ioi_task` — 間接賓語識別 / Indirect object identification

#### 參數 / Parameters

| 參數 / Parameter | 類型 / Type | 默認值 / Default | 描述 / Description |
|-----------------|------------|-----------------|-------------------|
| `task_name` | str | 必填 / Required | 注冊表中的任務名稱 / Name of task from registry |
| `task_config` | TaskConfig | None | 可選自定義配置 / Optional custom config |
| `model_name` | str | `"gpt2"` | HuggingFace 模型標識符 / HuggingFace model identifier |
| `checkpoint` | str | None | 模型檢查點 / Model checkpoint/revision |
| `num_samples` | int | `10` | 使用的示例數量 / Number of examples to use |
| `device` | str | `"auto"` | 設備 / Device: "auto", "cuda", or "cpu" |
| `layer_idx` | int | None | 提取的層 / Layer to extract from (None = last) |

#### 返回值 / Returns

返回一個 `TaskFunctionVec` 對象，包含 / A `TaskFunctionVec` object with:
- `task_name`：任務名稱 / Name of the task
- `function_vec`：提取的函數向量（numpy 數組）/ The extracted function vector (numpy array)
- `normalization`：使用的歸一化方法 / Normalization method used (default: "l2")

---

### 2. 高級接口 / Advanced Interface (Full Control)

高級接口提供對模型、注意力頭、採樣和提取參數的手動配置。適用於研究實驗、自定義任務和精細控制。

The advanced interface provides manual configuration of models, heads, sampling, and extraction parameters. Best for research experiments, custom tasks, and fine-tuned control.

#### 完整示例 / Complete Example

```python
from function_vecs.extract_function_vecs import (
    extract_task_function_vec,
    ExtractConfig,
    Headset,
    extract_informative_heads
)
from tasks.registry import get_task
from tasks.base_task import TaskConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 步驟 1：加載模型和分詞器 / Step 1: Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()

# 步驟 2：從注冊表獲取任務 / Step 2: Get task from registry
task_config = TaskConfig(
    name="basic_arithmetic",
    data_path="dataset/simple.csv",
    input_column="question",
    output_column="answer"
)
task = get_task("simple_arithmetic", task_config)

# 步驟 3：配置提取參數 / Step 3: Configure extraction parameters
config = ExtractConfig(
    model_name=model_name,
    device="cuda",
    num_samples_per_task=20,
    batch_size=8,
    layers=[11],                      # 分析的層 / Which layers to analyze
    topk_heads=10,                    # 選擇的注意力頭數量 / Number of heads to select
    head_selection="topk",            # "topk" 或 / or "soft"
    seed=42
)

# 步驟 4：選擇信息豐富的注意力頭 / Step 4: Select informative heads
# 選項 A：使用 AIE 自動選擇 / Option A: Automatic selection using AIE
headset = extract_informative_heads(config, [task])

# 選項 B：手動指定 / Option B: Manual specification
headset = Headset(
    mode="topk",
    heads=[(11, 0), (11, 1), (11, 2), (11, 5)]  # (層, 頭) 元組 / (layer, head) tuples
)

# 步驟 5：提取函數向量 / Step 5: Extract function vector
function_vec = extract_task_function_vec(
    task=task,
    config=config,
    head_set=headset,
    model=model,
    tokenizer=tokenizer
)
```

#### 創建自定義任務 / Creating Custom Tasks

```python
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# 使用內存數據創建任務 / Create task with custom in-memory data
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory",
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"},
        {"input": "hello", "output": "HELLO"}
    ],
    input_column="input",
    output_column="output"
)
task = get_task("simple_icl", task_config)

# 為自定義任務提取函數向量 / Extract function vector for custom task
function_vec = extract_task_function_vec(task, config, headset)
```

---

## 加載模型檢查點 / Loading Model Checkpoints

**OLMo-2** 和 **Crystal/CrystalCoder** 模型提供中間預訓練檢查點，允許分析不同訓練階段的模型行為。

Both **OLMo-2** and **Crystal/CrystalCoder** models provide intermediate pre-training checkpoints, allowing you to analyze model behavior at different stages of training.

### 支持的模型與檢查點 / Supported Models with Checkpoints

#### OLMo-2-1124-7B

**模型 ID / Model ID:** `allenai/OLMo-2-1124-7B`

**檢查點格式 / Checkpoint Format:** `stage1-stepXXXX-tokensYYYB`

**示例 / Examples:**
- `stage1-step1000-tokens5B` — 約 5B 令牌後的早期檢查點 / Early checkpoint after ~5B tokens
- `stage1-step10000-tokens42B` — 早期訓練檢查點 / Early training checkpoint
- `stage1-step100000-tokens420B` — 中期訓練檢查點 / Mid-training checkpoint
- `main`（默認）— 最終訓練模型 / (default) Final trained model

#### LLM360/Crystal

**模型 ID / Model ID:** `LLM360/Crystal`

**檢查點格式 / Checkpoint Format:** `CrystalCoder_phaseN_checkpoint_XXXXXX`

### 使用簡單接口 / Usage with Simple Interface

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 從 OLMo-2 早期檢查點提取 / Extract from OLMo-2 early checkpoint
function_vec = extract_function_vector_simple(
    task_name="basic_arithmetic",
    model_name="allenai/OLMo-2-1124-7B",
    checkpoint="stage1-step1000-tokens5B",
    num_samples=20,
    device="cuda"
)
```

### 分析訓練動態 / Analyzing Training Dynamics

```python
# OLMo-2 訓練進展 / OLMo-2 training progression
checkpoints = [
    "stage1-step1000-tokens5B",
    "stage1-step5000-tokens21B",
    "stage1-step10000-tokens42B"
]

function_vecs = []
for checkpoint in checkpoints:
    vec = extract_function_vector_simple(
        task_name="basic_arithmetic",
        model_name="allenai/OLMo-2-1124-7B",
        checkpoint=checkpoint,
        num_samples=20
    )
    function_vecs.append(vec)
    print(f"{checkpoint}: norm={vec.function_vec.dot(vec.function_vec):.6f}")
```

### 查找可用檢查點 / Finding Available Checkpoints

```python
from huggingface_hub import list_repo_refs

# 列出所有 OLMo-2 檢查點 / List all OLMo-2 checkpoints
out = list_repo_refs("allenai/OLMo-2-1124-7B")
checkpoints = [b.name for b in out.branches]
print(f"OLMo-2 checkpoints: {checkpoints}")
```

---

## 構建多任務技能基底 / Building a Skill Basis from Multiple Tasks

從多個任務提取函數向量，使用 SVD 創建共享技能基底，揭示任務間共享的底層「技能維度」。

Extract function vectors from multiple tasks and create a shared skill basis using SVD. This reveals the underlying "skill dimensions" shared across tasks.

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple,
    stack_function_vecs,
    build_skill_basis
)

# 步驟 1：提取多個任務的函數向量 / Step 1: Extract function vectors for multiple tasks
task_names = ["basic_arithmetic", "simple_icl", "token_reversal", "part_of_speech"]
function_vecs = []

for task_name in task_names:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(task_name, model_name="gpt2", num_samples=20)
    function_vecs.append(vec)

# 步驟 2：將向量堆疊成矩陣 / Step 2: Stack vectors into a matrix
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape (d_model x num_tasks): {task_matrix.V.shape}")

# 步驟 3：使用 SVD 構建技能基底 / Step 3: Build skill basis using SVD
skill_basis = build_skill_basis(
    task_matrix,
    method="svd",
    k=6  # 技能維度數量（-1 = 自動選擇至 95% 能量）/ Number of skill dimensions (-1 for auto)
)

print(f"Skill basis U shape: {skill_basis.U.shape}")  # (d_model, k)
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# 步驟 4：分析技能空間中的任務關係 / Step 4: Analyze task relationships in skill space
task_projections = skill_basis.Vt  # (k, num_tasks)
```

### 解讀技能基底 / Interpreting the Skill Basis

- **U**：模型空間中的技能基底向量 / Skill basis vectors in model space (d_model x k)
- **S**：表示每個技能維度重要性的奇異值 / Singular values indicating importance of each skill dimension
- **Vt**：技能維度上的任務載荷 / Task loadings on skill dimensions (k x num_tasks)

---

## 發現可用任務 / Discovering Available Tasks

### 從 Python / From Python

```python
from function_vecs.extract_function_vecs import discover_all_tasks

# 列出所有可用任務及描述 / List all available tasks with descriptions
task_names = discover_all_tasks()
```

### 從命令行 / From Command Line

```bash
python function_vecs/extract_function_vecs.py
```

### 使用任務注冊表 / Using Task Registry

```python
from tasks.registry import list_tasks, get_task_info

# 列出所有任務名稱 / List all task names
tasks = list_tasks()
print(f"Available tasks: {tasks}")

# 獲取特定任務的信息 / Get info about a specific task
info = get_task_info("simple_arithmetic")
print(info)
```

---

## 關鍵配置參數 / Key Configuration Parameters

### ExtractConfig

```python
@dataclass
class ExtractConfig:
    # 模型配置 / Model configuration
    model_name: str = "EleutherAI/gpt-j-6B"
    checkpoint: Optional[str] = None  # 模型檢查點 / Model checkpoint/revision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    seed: int = 42
    layers: Optional[List[int]] = None  # None = 使用所有層 / If None, use all layers

    # 采樣配置 / Sampling configuration
    num_samples_per_task: int = 20
    num_shuffled_controls_per_task: int = 10

    # 注意力頭選擇配置 / Head selection configuration
    head_selection: Literal["topk", "soft"] = "topk"
    topk_heads: int = 10
    cached_headset_path: Optional[str] = None

    # 基底配置 / Basis configuration
    basis_method: Literal["svd", "pca"] = "svd"
    basis_dim: int = 20
    eps: float = 0.01  # 用於 eps-rank / for eps-rank
```

### Headset

```python
@dataclass
class Headset:
    mode: Literal["topk", "soft"]  # 選擇模式 / Selection mode
    heads: List[Tuple[int, int]]    # (層, 頭) 元組列表 / List of (layer, head) tuples
    weights: Optional[np.ndarray]   # "soft" 模式的可選權重 / Optional weights for "soft" mode
```

**兩種模式 / Two modes:**
- `topk`：均等地對所選前 k 個頭的貢獻求和 / Sum contributions from top-k selected heads equally
- `soft`：使用提供的權重進行加權組合 / Weighted combination using provided weights

---

## 工作原理 / How It Works

函數向量提取流程 / The function vector extraction pipeline:

1. **采樣 ICL 提示 / Sample ICL Prompts**：從任務獲取上下文學習示例 / Get in-context learning examples from the task
2. **生成對照提示 / Generate Control Prompts**：創建打亂的對照示例 / Create shuffled control examples
3. **計算注意力頭重要性 / Compute Head Importance**：使用 AIE（注意力重要性估計）/ Uses AIE (Attention Importance Estimation)
4. **提取每頭貢獻 / Extract Per-Head Contributions**：計算每個注意力頭對殘差流的貢獻 / Compute how each attention head contributes to residual stream
5. **跨示例平均 / Average Across Examples**：每個頭在所有 ICL 示例中的平均貢獻 / Mean contribution of each head across all ICL examples
6. **折疊為函數向量 / Collapse to Function Vector**：對所選頭求和（或加權求和）/ Sum (or weighted sum) across selected heads
7. **歸一化 / Normalize**：應用 L2 歸一化 / Apply L2 normalization

### 數學細節 / Mathematical Details

對於層 l 中的注意力頭 h / For a given attention head h in layer l:
- O_h：輸出投影前的頭輸出 / Head's output before the output projection
- W_o^h：輸出投影的對應列 / Corresponding columns of the output projection
- 頭貢獻 / Head contribution: `c_h = O_h @ W_o^h`

函數向量 / Function vector: `v = normalize(Σ_h∈H c_h)`

---

## 運行測試 / Running Tests

```bash
# 運行所有測試 / Run all tests
pytest tests/test_simple_interface.py -v
pytest tests/test_function_vecs_revised.py -v
pytest tests/test_basic_icl_tasks.py -v
```

---

## 技巧與最佳實踐 / Tips and Best Practices

| 主題 / Topic | 建議 / Recommendation |
|-------------|----------------------|
| 模型選擇 / Model Selection | 測試用小模型：`distilgpt2`, `gpt2`；生產用大模型：`gpt2-medium`, `EleutherAI/gpt-j-6B` |
| 采樣數量 / Number of Samples | 快速實驗 10-20，研究用 50+ / 10-20 for quick experiments, 50+ for research |
| 層選擇 / Layer Selection | 最後一層語義信息最豐富，中間層捕捉更多句法模式 / Last layer has most semantic info, middle layers capture syntactic patterns |
| 注意力頭選擇 / Head Selection | 從 5-10 個頭開始，根據結果調整 / Start with 5-10 heads, adjust based on results |
| 設備管理 / Device Management | 使用 `device="auto"` 自動選擇 CUDA / Use `device="auto"` to automatically select CUDA |

---

## 故障排除 / Troubleshooting

| 問題 / Issue | 解決方案 / Solution |
|-------------|-------------------|
| CUDA 內存不足 / CUDA out of memory | 減少 `batch_size` 或使用較小模型 / Reduce `batch_size` or use smaller model |
| 任務未找到 / Task not found | 運行 `discover_all_tasks()` 查看可用任務 / Run `discover_all_tasks()` to see available tasks |
| 導入錯誤 / Import errors | 確保所有依賴已安裝 / Ensure all dependencies are installed |
| 函數向量形狀異常 / Unexpected vector shape | 檢查模型是否正確加載，`layer_idx` 是否有效 / Check model loaded correctly and `layer_idx` is valid |

---

## 示例庫 / Examples Gallery

### 示例 1：快速提取 / Example 1: Quick Extraction

```python
vec = extract_function_vector_simple("basic_arithmetic")
print(f"Extracted {vec.function_vec.shape[0]}-dimensional vector for {vec.task_name}")
```

### 示例 2：比較多個模型 / Example 2: Compare Multiple Models

```python
models = ["distilgpt2", "gpt2", "gpt2-medium"]
vectors = {}

for model_name in models:
    vec = extract_function_vector_simple("basic_arithmetic", model_name=model_name, num_samples=20)
    vectors[model_name] = vec.function_vec
```

### 示例 3：多任務分析 / Example 3: Multi-Task Analysis

```python
tasks = ["basic_arithmetic", "simple_icl", "token_reversal"]
vecs = [extract_function_vector_simple(t, num_samples=20) for t in tasks]

task_matrix = stack_function_vecs(vecs)
skill_basis = build_skill_basis(task_matrix, k=-1)  # 自動選擇 k / Auto-select k

print(f"Found {skill_basis.U.shape[1]} skill dimensions")
```

---

## 相關文檔 / Related Documentation

- [主項目 README / Main Project README](../README.md) | [中英對照版 / Bilingual](../README_ZH_EN.md)
- [任務系統文檔 / Task System Documentation](../tasks/README.md) | [中英對照版 / Bilingual](../tasks/README_ZH_EN.md)
- [模型評估腳本 / Model Evaluation Scripts](../scripts/)

---

> 📄 本文件為中英對照版本。英文原版請參閱 [README.md](README.md)。
> This is the bilingual version. For the original English documentation, see [README.md](README.md).
