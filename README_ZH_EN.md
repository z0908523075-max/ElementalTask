# ElementalTask — 中英對照文檔 / Bilingual Documentation

---

## 目錄 / Table of Contents

- [項目結構 / Project Structure](#項目結構--project-structure)
- [函數向量與基底提取 / Function Vector and Basis Extraction](#函數向量與基底提取--function-vector-and-basis-extraction)
- [開發待辦事項 / Developmental TODOs](#開發待辦事項--developmental-todos)
- [測試 / Test](#測試--test)

---

## 項目結構 / Project Structure

```
ElementalTask/
├── configs/                    # 模型檢查點配置 / Model checkpoint configurations
│   ├── olmo2_checkpoints.json  # OLMo-2 1B/7B 檢查點定義 / OLMo-2 1B/7B checkpoint definitions
│   └── ...
│
├── dataset/                    # 任務數據集（CSV 格式）/ Task datasets (CSVs)
│   ├── simple.csv              # 原子 ICL 任務 / Atomic ICL tasks (uppercase, lowercase, first_letter, etc.)
│   ├── simple_spaced.csv       # 帶間距的原子任務變體 / Spaced variant of atomic tasks
│   ├── compositional.csv       # 組合任務（連鎖操作）/ Compositional tasks (chained operations)
│   ├── compositional_spaced.csv
│   ├── math_expressions.csv    # 數學/算術任務 / Math/arithmetic tasks
│   ├── textfrct/               # TextFRCT 基準任務 / TextFRCT benchmark tasks
│   └── ...
│
├── tasks/                      # 任務框架 / Task framework
│   ├── base_task.py            # BaseTask 抽象類與 TaskConfig / BaseTask abstract class and TaskConfig
│   ├── registry.py             # 任務發現、注冊與列表 / Task discovery, registration, and listing
│   └── implementations/        # 任務實現 / Task implementations
│       ├── simple_icl_task.py          # 原子 ICL 任務（10 個類別）/ Atomic ICL tasks (10 categories)
│       ├── compositional_task.py       # 組合任務（連鎖操作）/ Compositional tasks (chained ops)
│       ├── textfrct_task.py            # TextFRCT 基準任務 / TextFRCT benchmark tasks
│       ├── math_task.py                # 數學表達式任務 / Math expression tasks
│       ├── basic_arithmetic_task.py    # 基礎算術 / Basic arithmetic
│       ├── copying_task.py             # 精確複製 / Exact copying
│       ├── token_reversal_task.py      # 令牌反轉 / Token reversal
│       ├── string_analogy_task.py      # 字符串類比 / String analogies
│       ├── ignoring_context_task.py    # 忽略無關上下文 / Ignoring irrelevant context
│       ├── ioi_task.py                 # 間接賓語識別 / Indirect Object Identification
│       └── part_of_speech_task.py      # 詞性標注 / Part of speech tagging
│
├── function_vecs/              # 函數向量提取與分析 / Function vector extraction & analysis
│   ├── extract_function_vecs.py    # 函數向量提取（簡單 + 詳細 API）/ FV extraction (simple + detailed API)
│   └── experiments/                # 函數向量分析實驗 / FV analysis experiments
│       └── analyze_real_tasks.py
├── scripts/                    # 評估與分析腳本 / Evaluation & analysis scripts
│   ├── eval_across_checkpoints.py          # 主檢查點評估 / Main checkpoint evaluation
│   ├── eval_array_job_v2.sh                # SLURM 批量評估作業 / SLURM array job for batch eval
│   ├── eval_new_compositions.sh            # 新組合任務評估 / Eval for new compositional tasks
│   ├── eval_checkpoints.sh                 # 單節點評估腳本 / Single-node eval script
│   ├── run_unified_eval.py                 # 統一評估運行器 / Unified evaluation runner
│   └── trajectory_analysis/                # 發展軌跡分析 / Developmental trajectory analysis
│       └── predict_compositional_from_components.py
│
├── analysis/                   # 繪圖與可視化 / Plotting & visualization
├── results/                    # 評估輸出 / Evaluation outputs
├── plots/                      # 生成的圖表 / Generated figures
├── logs/                       # SLURM 作業日誌 / SLURM job logs
└── tests/                      # 單元測試 / Unit tests
    └── test_basic_icl_tasks.py
```

---

## 過去記錄 / Past Notes

**支持的模型 / Supported Models**
- Olmo 1
- Olmo 2
- LLM360
- Emmy's Adhoc ckpts

**任務（第一階段 - 完整性測試）/ Tasks (phase I - sanity check)**

- 精確複製 / 語義複製（Exact copying / semantic copying）
  - 複製輸入 / Copy the input:
    - 輸入 / Input: xyzabc
- 算術 / Arithmetic
- 同義詞 / 反義詞（Synonyms / antonyms）
- 並行結構 / 模板（Parallel structures / templates）
- 反轉 / 令牌操作（Reversal / token ops）
  - 反轉單詞 cat：tac / Reverse the word cat: tac
- 事實回憶 / Factual recall
  - facebook/kilt_tasks

**複雜任務：如何用基礎任務構建複雜任務？**
**Complex Tasks: How can we construct a complex task with the elemental tasks we proposed?**
- 抽取式問答 = 理解 + 推理 + 精確複製 / extractiveQA = Understanding + Reasoning + Exact Copy
- 開放域問答 = 記憶 + 理解 + 推理 + 語言學 / Open domain QA = Memorization + Understanding + Reasoning + Linguistics
- 自然語言推斷 = 理解 + 推理 / Natural Language Inference = Understanding + Reasoning

---

## 使用示例 / Usage Examples

### 列出所有可用檢查點 / List all available checkpoints

```bash
python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --list_checkpoints_only
```

### 評估 10 個均勻採樣的檢查點 / Evaluate 10 uniformly sampled checkpoints

```bash
python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --use_vllm
```

### 評估特定檢查點 / Evaluate specific checkpoints

```bash
python scripts/measure_ckpt_interp_perf.py \
  --model_id allenai/OLMo-1B-hf \
  --checkpoints step10000-tokens42B step50000-tokens210B \
  --use_vllm
```

### 生成圖表 / Generate images

```bash
# 生成所有圖表類型 / Generate all plot types
python analysis/plotting.py --csv_path output/olmo2_ckpt_interp_results_firstckpts.csv

# 僅生成性能曲線 / Generate only performance curves
python analysis/plotting.py \
  --csv_path output/olmo2_ckpt_interp_results.csv \
  --plot_type curves

# 生成分組任務曲線 / Generate grouped task curves
python analysis/plotting.py \
  --csv_path output/olmo2_ckpt_interp_results.csv \
  --plot_type grouped

# 自定義輸出目錄和圖表大小 / Custom output directory and figure size
python analysis/plotting.py \
  --csv_path output/olmo2_ckpt_interp_results.csv \
  --output_dir my_plots \
  --figsize 16 10
```

---

## 函數向量與基底提取 / Function Vector and Basis Extraction

函數向量提取的代碼存放在 `function_vecs` 目錄中。注意：此部分代碼假設任務具有合理的 ICL 表示。

The code to extract function vectors from tasks and form a basis lives in the dir `function_vecs`. Note that the code in this section assumes that the task has an ICL representation that makes sense.

函數向量是一個 L2 歸一化向量，對應於排名前 k 個最具信息量的注意力頭，在 ICL 示例與對照示例（輸入→輸出映射被打亂的示例）之間，如何在殘差流中產生不同貢獻。

The function vector is an L2-normalized vector which corresponds to how the top-k most informative attention heads contribute differently in the residual stream across ICL examples versus control examples (ones where the input -> output mapping is shuffled).

### 簡單與複雜提取接口 / Simple vs Complex Extraction

我們提供兩種函數向量提取接口：

We provide two interfaces for function vector extraction:

**簡單接口（Simple Interface）** — 快速提取的一站式函數 / One-stop function for quick extraction:
- 自動配置模型加載、注意力頭選擇和提取參數 / Automatically configures model loading, head selection, and extraction parameters
- 適用於：快速原型設計、標準用例、入門 / Best for: Rapid prototyping, standard use cases, getting started

**詳細接口（Detailed Interface）** — 對提取流程的完全控制 / Full control over extraction pipeline:
- 手動配置模型、注意力頭、采樣和提取參數 / Manual configuration of models, heads, sampling, and extraction parameters
- 適用於：研究實驗、自定義任務、精細控制 / Best for: Research experiments, custom tasks, fine-tuned control

### 快速開始：簡化接口 / Quick Start: Simplified Interface

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 簡單用法 - 使用所有默認值 / Simple usage - uses all defaults
function_vec = extract_function_vector_simple("simple_icl", num_samples=10)
print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")

# 使用自定義模型和設備 / With custom model and device
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="distilgpt2",
    num_samples=5,
    device="cpu",
    layer_idx=5  # 使用特定層 / Use specific layer
)
```

注冊表中的可用任務 / Available tasks in registry: `["simple_icl", "simple", "textfrct", "math"]`

### 高級用法：詳細接口 / Advanced Usage: Detailed Interface

```python
from function_vecs.extract_function_vecs import extract_task_function_vec, ExtractConfig, Headset
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# 1. 定義任務或導入現有任務 / Define your task or import one

# 使用內存數據創建任務 / Create task with custom in-memory data
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory",
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"}
    ]
)
task_from_config = get_task("simple_icl", task_config)

# 使用注冊表中的現有任務 / Use existing task from registry
simple_icl_config = TaskConfig(
    name="simple_icl",
    data_path="dataset/simple.csv",
    input_column="question",
    output_column="answer"
)
task = get_task("simple_icl", simple_icl_config)

# 2. 配置提取 / Configure extraction
extract_config = ExtractConfig(
    model_name="gpt2",
    num_samples_per_task=10,
    batch_size=4,
    layers=[11],  # 聚焦特定層 / Focus on specific layers
    device="auto"
)

# 3. 創建注意力頭集合 / Create headset
head_set = Headset(mode="topk", heads=[(11, 0), (11, 1), (11, 2)])

# 4. 提取函數向量 / Extract function vectors
function_vec_from_config = extract_task_function_vec(task_from_config, extract_config, head_set)
function_vec_from_task = extract_task_function_vec(task, extract_config, head_set)
```

### 從多個任務構建基底 / Forming a basis from multiple tasks

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple,
    stack_function_vecs,
    build_skill_basis
)

# 提取多個任務的函數向量 / Extract function vectors for multiple tasks
tasks = ["simple_icl"]
function_vecs = []

for task_name in tasks:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(task_name, num_samples=10)
    function_vecs.append(vec)

# 將向量堆疊成矩陣 / Stack vectors into a matrix
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape: {task_matrix.V.shape}")

# 使用 SVD 構建技能基底 / Build skill basis using SVD
skill_basis = build_skill_basis(task_matrix, method="svd", k=6)
print(f"Skill basis dimensions: {skill_basis.U.shape}")
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# 分析技能空間中的任務關係 / Analyze task relationships in the skill space
task_projections = skill_basis.Vt
print(f"Task projections shape: {task_projections.shape}")
```

---

## 開發待辦事項 / Developmental TODOs

**數據準備 / Data Preparation**
- 將所有數據寫入 `data` 目錄，以任務名稱為子目錄 / Write everything to `data` dir, with the task name as the sub-directory
- 通用數據格式 / A universal data format
  - 本地存儲為 `HF dataset .jsonl`，`lm_input` 表示語言模型輸入，`reference` 為預期輸出

**模型推理 / Model Inference**
- 傳入模型名稱、檢查點 / Pass model name, checkpoint
- 運行推理（寫入 `outputs`）/ 評估 / Run inference (write to `outputs`) / evaluation

**數據保存 / Data Saving**
- Kaiser 在某個保存目錄中完成 / Kaiser made in save directory somewhere

---

## 測試 / Test

```bash
python models/evaluate_models.py \
  --model_id LLM360/Crystal \
  --max_new_tokens 5 \
  --chkpt main

python models/evaluate_models.py \
  --task_name FRCT_CV1_ScrambledWords \
  --max_new_tokens 5 \
  --chkpt main
```

---

> 📄 本文件為中英對照版本。英文原版請參閱 [README.md](README.md)。
> This is the bilingual version. For the original English documentation, see [README.md](README.md).
