# Function Vectors 擷取

此目錄包含用於從 language models 擷取 **function vectors** 的工具。Function vectors 是經過 L2-normalized 的向量，可捕捉 attention heads 對 residual stream 的貢獻在 in-context learning (ICL) 範例與 control 範例（input→output 對應被打亂）之間的差異。它們代表模型學到的任務特定資訊。

## 目錄

- [快速開始](#快速開始)
- [兩種擷取介面](#兩種擷取介面)
  - [簡易介面（建議）](#1-簡易介面建議)
  - [進階介面](#2-進階介面完整控制)
- [載入模型 Checkpoints](#載入模型-checkpoints)
- [從多個任務建立 Skill Basis](#從多個任務建立-skill-basis)
- [探索可用任務](#探索可用任務)
- [關鍵設定參數](#關鍵設定參數)
- [運作方式](#運作方式)
- [執行測試](#執行測試)

---

## 快速開始

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 用一行擷取 function vector
function_vec = extract_function_vector_simple("basic_arithmetic", num_samples=10)

print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")
print(f"L2 norm: {function_vec.function_vec.dot(function_vec.function_vec):.6f}")  # ~1.0
```

---

## 兩種擷取介面

### 1. 簡易介面（建議）

簡易介面提供自動設定的一站式函式擷取。最適合快速原型、標準使用情境與入門使用。

#### 基本用法

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 最精簡用法 - 使用所有預設值
function_vec = extract_function_vector_simple("basic_arithmetic")

# 使用自訂參數
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="gpt2",          # 或 "distilgpt2", "EleutherAI/gpt-j-6B" 等
    num_samples=20,              # 使用的範例數量
    device="cuda",               # "auto", "cuda", 或 "cpu"
    layer_idx=11                 # 特定層（None = 使用最後一層）
)

print(f"Task: {function_vec.task_name}")
print(f"Shape: {function_vec.function_vec.shape}")
print(f"Normalization: {function_vec.normalization}")  # "l2"
```

#### 可用任務

任務會自動從 task registry 探索。可用任務包括：
- `basic_arithmetic` - 基本算術運算
- `simple_icl` - 簡單的 in-context learning 任務
- `simple` - 簡單任務範例
- `textfrct` - TextFRCT 資料集任務
- `part_of_speech` - 詞性識別
- `token_reversal` - token 反轉操作
- `math` - 數學推理任務
- `ioi_task` - 間接受詞識別

請參閱[探索可用任務](#探索可用任務)以動態列出所有任務。

#### 參數

| 參數 | 類型 | 預設值 | 說明 |
|-----------|------|---------|-------------|
| `task_name` | str | 必填 | 來自 registry 的任務名稱 |
| `task_config` | TaskConfig | None | 可選的自訂設定 |
| `model_name` | str | `"gpt2"` | HuggingFace model identifier |
| `checkpoint` | str | None | 模型 checkpoint/revision（例如："stage1-step1000-tokens5B"） |
| `num_samples` | int | `10` | 使用的範例數量 |
| `device` | str | `"auto"` | 裝置：「auto」、「cuda」或「cpu」 |
| `layer_idx` | int | None | 要擷取的層（None = 最後一層） |

#### 回傳內容

一個 `TaskFunctionVec` 物件，包含：
- `task_name`：任務名稱
- `function_vec`：擷取出的 function vector（numpy array）
- `normalization`：使用的 normalization 方法（預設："l2"）

---

### 2. 進階介面（完整控制）

進階介面提供模型、heads、取樣與擷取參數的手動設定。最適合研究實驗、自訂任務與精細化控制。

#### 完整範例

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

# 步驟 1：載入模型與 tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()

# 步驟 2：從 registry 取得任務
task_config = TaskConfig(
    name="basic_arithmetic",
    data_path="dataset/simple.csv",  # 可選：自訂資料路徑
    input_column="question",
    output_column="answer"
)
task = get_task("simple_arithmetic", task_config)

# 步驟 3：設定擷取參數
config = ExtractConfig(
    model_name=model_name,
    device="cuda",
    num_samples_per_task=20,
    batch_size=8,
    layers=[11],                      # 要分析哪些層
    topk_heads=10,                    # 要選取的 heads 數量
    head_selection="topk",            # "topk" 或 "soft"
    seed=42
)

# 步驟 4：選取具資訊量的 heads
# 選項 A：使用 AIE（Attention Importance Estimation）自動選取
headset = extract_informative_heads(config, [task])
print(f"Selected heads: {headset.heads}")

# 選項 B：手動指定
headset = Headset(
    mode="topk",
    heads=[(11, 0), (11, 1), (11, 2), (11, 5)]  # (layer, head) tuple
)

# 步驟 5：擷取 function vector
function_vec = extract_task_function_vec(
    task=task,
    config=config,
    head_set=headset,
    model=model,
    tokenizer=tokenizer
)

print(f"Task: {function_vec.task_name}")
print(f"Function vector shape: {function_vec.function_vec.shape}")
```

#### 建立自訂任務

你可以使用記憶體內資料建立任務：

```python
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# 使用自訂記憶體內資料建立任務
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
task = get_task("simple_icl", task_config)  # 使用 simple_icl 作為基底類別

# 現在為這個自訂任務擷取 function vector
function_vec = extract_task_function_vec(task, config, headset)
```

---

## 載入模型 Checkpoints

**OLMo-2** 與 **Crystal/CrystalCoder** 模型都提供中間 pre-training checkpoints，讓你可以分析模型在不同訓練階段的行為。

### 支援提供 Checkpoints 的模型

#### OLMo-2-1124-7B

**Model ID:** `allenai/OLMo-2-1124-7B`

**Checkpoint 格式：** `stage1-stepXXXX-tokensYYYB`

**範例：**
- `stage1-step1000-tokens5B` - 約 5B tokens 後的早期 checkpoint
- `stage1-step10000-tokens42B` - 早期訓練 checkpoint
- `stage1-step100000-tokens420B` - 訓練中期 checkpoint
- `main`（預設）- 最終訓練完成模型

#### LLM360/Crystal

**Model ID:** `LLM360/Crystal`

**Checkpoint 格式：** `CrystalCoder_phaseN_checkpoint_XXXXXX`

**範例：**
- `CrystalCoder_phase1_checkpoint_055500` - Phase 1 checkpoint
- `CrystalCoder_phase3_checkpoint_027728` - 預設／最終 checkpoint

### 搭配簡易介面的用法

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 從 OLMo-2 早期 checkpoint 擷取
function_vec = extract_function_vector_simple(
    task_name="basic_arithmetic",
    model_name="allenai/OLMo-2-1124-7B",
    checkpoint="stage1-step1000-tokens5B",  # 指定 checkpoint
    num_samples=20,
    device="cuda"
)

# 從 Crystal Phase 1 checkpoint 擷取
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="LLM360/Crystal",
    checkpoint="CrystalCoder_phase1_checkpoint_055500",
    num_samples=20,
    device="cuda"
)
```

### 搭配進階介面的用法

```python
from function_vecs.extract_function_vecs import ExtractConfig, extract_task_function_vec
from tasks.registry import get_task
from tasks.base_task import TaskConfig

# 設定包含 checkpoint 的擷取參數
config = ExtractConfig(
    model_name="allenai/OLMo-2-1124-7B",
    checkpoint="stage1-step1000-tokens5B",  # 指定 checkpoint
    device="cuda",
    num_samples_per_task=20,
    layers=[31]  # 最後一層
)

# 載入任務並擷取
task_config = TaskConfig(name="basic_arithmetic")
task = get_task("basic_arithmetic", task_config)
function_vec = extract_task_function_vec(config, task)
```

### 分析訓練動態

比較不同訓練 checkpoints 的 function vectors：

```python
# OLMo-2 訓練進程
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

# 分析 function vectors 如何隨訓練演變
# （例如 checkpoints 之間的 cosine similarity）
```

### 尋找可用的 Checkpoints

```python
from huggingface_hub import list_repo_refs

# 列出所有 OLMo-2 checkpoints
out = list_repo_refs("allenai/OLMo-2-1124-7B")
checkpoints = [b.name for b in out.branches]
print(f"OLMo-2 checkpoints: {checkpoints}")

# 造訪 HuggingFace 上 Crystal 的 "Files and versions" 分頁：
# https://huggingface.co/LLM360/Crystal/tree/main
```

### 注意事項

- 所有 checkpoint 載入都需要 `trust_remote_code=True`
- Checkpoints 會由 HuggingFace Hub 緩存在本機
- 如果未指定 checkpoint，則會載入 main／預設 branch
- 這兩個模型都受到 function_vecs 框架完整支援

---

## 從多個任務建立 Skill Basis

從多個任務擷取 function vectors，並使用 SVD 建立共享的 skill basis。這會揭示各任務共享的底層「skill dimensions」。

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple,
    stack_function_vecs,
    build_skill_basis
)

# 步驟 1：為多個任務擷取 function vectors
task_names = ["basic_arithmetic", "simple_icl", "token_reversal", "part_of_speech"]
function_vecs = []

for task_name in task_names:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(
        task_name,
        model_name="gpt2",
        num_samples=20
    )
    function_vecs.append(vec)

# 步驟 2：將向量堆疊成矩陣
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape (d_model x num_tasks): {task_matrix.V.shape}")
print(f"Tasks: {task_matrix.task_names}")

# 步驟 3：使用 SVD 建立 skill basis
skill_basis = build_skill_basis(
    task_matrix,
    method="svd",
    k=6  # skill 維度數量（-1 表示根據 95% energy 自動選取）
)

print(f"Skill basis U shape: {skill_basis.U.shape}")  # (d_model, k)
print(f"Singular values: {skill_basis.S}")
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# 步驟 4：在 skill 空間中分析任務關係
task_projections = skill_basis.Vt  # (k, num_tasks)
print(f"Task projections shape: {task_projections.shape}")

# Vt 的每一欄表示各任務在每個 skill 維度上的載荷程度
for i, task_name in enumerate(skill_basis.task_names):
    print(f"{task_name}: {skill_basis.Vt[:, i]}")
```

### 解讀 Skill Basis

- **U**：模型空間中的 skill basis 向量（d_model x k）
- **S**：表示各 skill 維度重要性的 singular values
- **Vt**：任務在 skill 維度上的載荷（k x num_tasks）

Skill basis 揭示了：
1. 哪些底層能力會在任務間共享
2. 任務在 skill 空間中彼此如何關聯
3. 「skill manifold」的維度

---

## 探索可用任務

### 從 Python

```python
from function_vecs.extract_function_vecs import discover_all_tasks

# 列出所有可用任務及其描述
task_names = discover_all_tasks()
```

### 從命令列

```bash
python function_vecs/extract_function_vecs.py
```

### 使用 Task Registry

```python
from tasks.registry import list_tasks, get_task_info

# 列出所有任務名稱
tasks = list_tasks()
print(f"Available tasks: {tasks}")

# 取得所有任務的詳細資訊
task_info = get_task_info()
for task_name, info in task_info.items():
    print(f"{task_name}: {info['class']} - {info['docstring'][:100]}")

# 取得特定任務的資訊
info = get_task_info("simple_arithmetic")
print(info)
```

---

## 關鍵設定參數

### ExtractConfig

擷取流程的完整設定：

```python
@dataclass
class ExtractConfig:
    # 模型設定
    model_name: str = "EleutherAI/gpt-j-6B"
    checkpoint: Optional[str] = None  # 模型 checkpoint/revision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    seed: int = 42
    layers: Optional[List[int]] = None  # 若為 None，使用所有層

    # 取樣設定
    num_samples_per_task: int = 20
    num_shuffled_controls_per_task: int = 10

    # Head selection 設定
    head_selection: Literal["topk", "soft"] = "topk"
    topk_heads: int = 10
    cached_headset_path: Optional[str] = None

    # Basis 設定
    basis_method: Literal["svd", "pca"] = "svd"
    basis_dim: int = 20
    eps: float = 0.01  # 用於 eps-rank
```

### Headset

指定要分析哪些 attention heads：

```python
@dataclass
class Headset:
    mode: Literal["topk", "soft"]  # 選取模式
    heads: List[Tuple[int, int]]    # (layer, head) tuple 清單
    weights: Optional[np.ndarray]   # "soft" 模式的可選權重
```

**兩種模式：**
- `topk`：對選出的 top-k heads 平等加總其貢獻
- `soft`：使用提供的權重進行加權組合

---

## 運作方式

function vector 擷取流程如下：

1. **取樣 ICL Prompts**：從任務取得 in-context learning 範例
   - 使用任務的 test split
   - 使用任務特定模板格式化 prompts

2. **產生 Control Prompts**：建立打亂的 control 範例
   - 透過打亂來破壞 input→output 對應
   - 格式與 ICL prompts 相同，但映射不正確

3. **計算 Head Importance**（若使用 `extract_informative_heads`）：
   - 使用 AIE（Attention Importance Estimation）
   - 衡量每個 head 被 control 替換後的效能下降
   - 選出 top-k 最具資訊量的 heads

4. **擷取各 Head 的貢獻**：
   - 對每個 prompt，計算各 attention head 如何對 residual stream 產生貢獻
   - 使用 hooks 擷取投影前／後的 tensors
   - 將 output projection 分解為各 head 的貢獻

5. **跨範例取平均**：
   - 對所有 ICL 範例取每個 head 的平均貢獻
   - 得到 `(d_model, num_heads)` 矩陣

6. **壓縮為 Function Vector**：
   - 對選定的 heads 加總（或加權加總）
   - 得到單一 d_model 維度向量

7. **正規化**：
   - 套用 L2 normalization（預設）
   - 最終的 function vector 具有單位範數

### 數學細節

對於第 l 層中的某個 attention head h：
- 令 O_h 為該 head 在 output projection 前的輸出
- 令 W_o^h 為對應 output projection 的欄位
- Head contribution：c_h = O_h @ W_o^h

Function vector:
```
v = normalize(Σ_h∈H c_h)
```

其中 H 為選定 heads 的集合。

---

## 執行測試

測試套件會驗證這兩種介面：

```bash
# 執行所有測試
pytest tests/test_simple_interface.py -v
pytest tests/test_function_vecs_revised.py -v
pytest tests/test_basic_icl_tasks.py -v

# 執行特定測試
pytest tests/test_simple_interface.py::test_simple_interface_existing_task -v

# 顯示輸出執行
python tests/test_simple_interface.py
```

### 測試範例

測試檔案提供可運作的範例：

**Simple Interface** ([tests/test_simple_interface.py](../tests/test_simple_interface.py)):
```python
# 基本用法
function_vec = extract_function_vector_simple(
    task_name="simple_arithmetic",
    model_name="distilgpt2",
    num_samples=3,
    device="cpu"
)

# 驗證其為 L2 normalized
assert abs(function_vec.function_vec.dot(function_vec.function_vec) - 1.0) < 1e-5
```

**Advanced Interface** ([tests/test_function_vecs_revised.py](../tests/test_function_vecs_revised.py)):
- 完整擷取流程
- 自訂 head 選取
- 多任務與 basis 建構

---

## 架構

### 關鍵檔案

- [`extract_function_vecs.py`](extract_function_vecs.py) - 主要擷取邏輯
- [`model_internal_getters.py`](model_internal_getters.py) - 模型架構內省
- [`activation_patching.py`](activation_patching.py) - activation intervention 工具

### 關鍵類別

- `ExtractConfig` - 擷取設定
- `Headset` - 要分析的 attention heads 規格
- `TaskFunctionVec` - 擷取後 function vector 的容器
- `TaskMatrix` - 多個 function vectors 的堆疊
- `SkillBasis` - 基於 SVD 的 skill 空間表示

### 關鍵函式

- `extract_function_vector_simple()` - 簡易一站式介面
- `extract_task_function_vec()` - 具完整控制能力的進階擷取
- `extract_informative_heads()` - 使用 AIE 自動選取 heads
- `stack_function_vecs()` - 將向量組合成矩陣
- `build_skill_basis()` - 透過 SVD 建構 skill basis

---

## 提示與最佳實踐

### 模型選擇
- 測試時先從較小模型開始：`distilgpt2`、`gpt2`
- 正式使用時改用較大模型：`gpt2-medium`、`gpt2-large`、`EleutherAI/gpt-j-6B`

### 範例數量
- 更多範例 = 更穩定的估計
- 建議：快速實驗使用 10-20，研究用途使用 50+
- 權衡：計算時間 vs. 穩定性

### 層選擇
- 最後一層通常具有最多語意資訊
- 中間層可能捕捉更多語法模式
- 針對你的任務實驗不同層

### Head 選擇
- `extract_informative_heads()` 會自動找出重要的 heads
- 當你具有領域知識時，手動選取會很有幫助
- 可先從 5-10 個 heads 開始，再依結果調整

### 裝置管理
- 使用 `device="auto"` 可在可用時自動選擇 CUDA
- 對於大型模型，請確保 GPU 記憶體足夠
- CPU 可用，但速度較慢

### Batch Size
- 較大的 batch = 更快，但需要更多記憶體
- 依你的 GPU 記憶體與模型大小調整
- 大多數情況下預設值 8 表現良好

---

## 疑難排解

### 問題：CUDA 記憶體不足
**解法**：降低 `batch_size` 或改用較小模型

### 問題：找不到任務
**解法**：執行 `discover_all_tasks()` 查看可用任務，或檢查任務名稱拼字

### 問題：Import 錯誤
**解法**：確認所有相依套件已安裝，且任務實作有效

### 問題：Function vector 的 shape 非預期
**解法**：檢查模型是否正確載入，以及 layer_idx 是否有效

---

## 範例集

### 範例 1：快速擷取
```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

vec = extract_function_vector_simple("basic_arithmetic")
print(f"Extracted {vec.function_vec.shape[0]}-dimensional vector for {vec.task_name}")
```

### 範例 2：比較多個模型
```python
models = ["distilgpt2", "gpt2", "gpt2-medium"]
vectors = {}

for model_name in models:
    vec = extract_function_vector_simple(
        "basic_arithmetic",
        model_name=model_name,
        num_samples=20
    )
    vectors[model_name] = vec.function_vec

# 比較向量（cosine similarity 等）
```

### 範例 3：多任務分析
```python
tasks = ["basic_arithmetic", "simple_icl", "token_reversal"]
vecs = [extract_function_vector_simple(t, num_samples=20) for t in tasks]

# 建立 skill basis
task_matrix = stack_function_vecs(vecs)
skill_basis = build_skill_basis(task_matrix, k=-1)  # 自動選取 k

print(f"Found {skill_basis.U.shape[1]} skill dimensions")
print(f"Capturing {skill_basis.S.sum():.2%} of variance")
```

---

## 引用

如果你在研究中使用此程式碼，請引用：

```bibtex
@software{elemental_tasks_function_vecs,
  title={Function Vector Extraction for Language Models},
  author={[Your Name/Team]},
  year={2025},
  url={https://github.com/[your-repo]/ElementalTask}
}
```

---

## 相關文件

- [主專案 README](../README.md)
- [任務系統文件](../tasks/README.md)
- [模型評估腳本](../scripts/)

---
