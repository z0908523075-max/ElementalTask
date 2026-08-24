## 專案結構

```
ElementalTask/
├── configs/                    # 模型 checkpoint 設定
│   ├── olmo2_checkpoints.json  # OLMo-2 1B/7B checkpoint 定義
│   └── ...
│
├── dataset/                    # 任務資料集 (CSVs)
│   ├── simple.csv              # 原子 ICL 任務（uppercase、lowercase、first_letter 等）
│   ├── simple_spaced.csv       # 原子任務的 spaced 變體
│   ├── compositional.csv       # 組合式任務（鏈式操作）
│   ├── compositional_spaced.csv
│   ├── math_expressions.csv    # 數學／算術任務
│   ├── textfrct/               # TextFRCT 基準任務
│   └── ...
│
├── tasks/                      # 任務框架
│   ├── base_task.py            # BaseTask 抽象類別與 TaskConfig
│   ├── registry.py             # 任務發現、註冊與列出
│   └── implementations/        # 任務實作
│       ├── simple_icl_task.py          # 原子 ICL 任務（10 個類別）
│       ├── compositional_task.py       # 組合式任務（鏈式操作）
│       ├── textfrct_task.py            # TextFRCT 基準任務
│       ├── math_task.py                # 數學表達式任務
│       ├── basic_arithmetic_task.py    # 基本算術
│       ├── copying_task.py             # 精確複製
│       ├── token_reversal_task.py      # token 反轉
│       ├── string_analogy_task.py      # 字串類比
│       ├── ignoring_context_task.py    # 忽略無關上下文
│       ├── ioi_task.py                 # Indirect Object Identification
│       └── part_of_speech_task.py      # 詞性標註
│
├── function_vecs/              # Function vector 擷取與分析
│   ├── extract_function_vecs.py    # FV 擷取（簡易 + 詳細 API）
│   └── experiments/                # FV 分析實驗
│       └── analyze_real_tasks.py
├── scripts/                    # 評估與分析腳本
│   ├── eval_across_checkpoints.py          # 主要 checkpoint 評估
│   ├── eval_array_job_v2.sh                # 用於批次評估的 SLURM array job
│   ├── eval_new_compositions.sh            # 新組合式任務的評估
│   ├── eval_checkpoints.sh                 # 單節點評估腳本
│   ├── run_unified_eval.py                 # 統一評估執行器
│   └── trajectory_analysis/                # 發展軌跡分析
│       └── predict_compositional_from_components.py
│
├── analysis/                   # 繪圖與視覺化
├── results/                    # 評估輸出
├── plots/                      # 產生的圖表
├── logs/                       # SLURM 工作日誌
└── tests/                      # 單元測試
    └── test_basic_icl_tasks.py
```

## 過往筆記
    - Olmo 1
    - Olmo 2
    - LLM360
    - Emmy 的臨時 ckpts

* 任務（階段 I - 健全性檢查）

    - 精確複製／語義複製
        - 複製輸入內容：
            Input: xyzabc
    - 算術
    - 同義詞／反義詞
    - 平行結構／模板
    - 反轉／token 操作
        - 將單字 cat 反轉：tac
    - 事實回憶
        - facebook/kilt_tasks

* 複雜任務：我們如何使用所提出的元素任務來構造複雜任務？
    - extractiveQA = 理解 + 推理 + 精確複製
    - Opendomain QA = 記憶 + 理解 + 推理 + 語言學
    - Natural Langauge Inference = 理解 + 推理


```
使用範例：
  # 列出所有可用的 checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --list_checkpoints_only

  # 評估 10 個均勻取樣的 checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --use_vllm

  # 評估指定的 checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --checkpoints step10000-tokens42B step50000-tokens210B --use_vllm
```





使用範例（產生圖片）：
```
  # 產生所有圖表類型
  python analysis/plotting.py --csv_path output/olmo2_ckpt_interp_results_firstckpts.csv

  # 只產生效能曲線
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type curves

  # 產生分組任務曲線
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type grouped

  # 自訂輸出目錄與圖形尺寸
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --output_dir my_plots --figsize
   16 10
```

### Function vector 與 basis 擷取

用於從任務中擷取 function vectors 並形成 basis 的程式碼位於 `function_vecs` 目錄中。請注意，本節中的程式碼假設該任務具有合理的 ICL 表示方式。

function vector 是經過 L2-normalized 的向量，對應於 top-k 最具資訊量的 attention heads 在 residual stream 中，對 ICL 範例與 control 範例（input -> output 對應被打亂的範例）產生不同貢獻的方式。

#### 簡易版與進階版擷取

我們提供兩種 function vector 擷取介面：

**簡易介面** - 用於快速擷取的一站式函式：
- 自動配置模型載入、head 選擇與擷取參數
- 可搭配既有 task registry 使用，並提供合理的預設值
- 最適合：快速原型、標準使用情境、入門使用

**詳細介面** - 完整控制擷取流程：
- 手動設定模型、heads、取樣與擷取參數  
- 支援自訂 task 設定與進階 head 選擇策略
- 最適合：研究實驗、自訂任務、精細化控制

#### 快速開始：簡化介面

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# 簡單用法 - 使用所有預設值
function_vec = extract_function_vector_simple("simple_icl", num_samples=10)
print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")

# 使用自訂模型與裝置
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="distilgpt2",
    num_samples=5,
    device="cpu",
    layer_idx=5  # 使用特定層
)
```

registry 中可用的任務：`["simple_icl", "simple", "textfrct", "math"]`

#### 進階用法：詳細介面

```python
from function_vecs.extract_function_vecs import extract_task_function_vec, ExtractConfig, Headset
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# 1. 定義你的任務或匯入既有任務

# 使用自訂記憶體內資料建立任務
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory", 
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"}
    ]
)
task_from_config = get_task("simple_icl", task_config)  # 使用 simple_icl 作為基底類別

# 使用 registry 中既有的任務
simple_icl_config = TaskConfig(
    name="simple_icl", 
    data_path="dataset/simple.csv",
    input_column="question",
    output_column="answer"
)
task = get_task("simple_icl", simple_icl_config)

# 2. 設定擷取參數
extract_config = ExtractConfig(
    model_name="gpt2",
    num_samples_per_task=10,
    batch_size=4,
    layers=[11],  # 聚焦於特定層
    device="auto"
)

# 3. 建立 headset（指定要分析哪些 attention heads）
head_set = Headset(mode="topk", heads=[(11, 0), (11, 1), (11, 2)])

# 4. 擷取 function vectors
function_vec_from_config = extract_task_function_vec(task_from_config, extract_config, head_set)
print(f"Custom task: {function_vec_from_config}")

function_vec_from_task = extract_task_function_vec(task, extract_config, head_set)
print(f"Registry task: {function_vec_from_task}")
```

#### 從多個任務形成 basis

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple, 
    stack_function_vecs, 
    build_skill_basis
)

# 為多個任務擷取 function vectors
tasks = ["simple_icl"]  # 可隨著更多任務可用而加入
function_vecs = []

for task_name in tasks:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(task_name, num_samples=10)
    function_vecs.append(vec)

# 將向量堆疊成矩陣
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape: {task_matrix.V.shape}")

# 使用 SVD 建立 skill basis
skill_basis = build_skill_basis(task_matrix, method="svd", k=6)
print(f"Skill basis dimensions: {skill_basis.U.shape}")
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# 在 skill 空間中分析任務關係
task_projections = skill_basis.Vt  # 投影到 skill 維度上的任務
print(f"Task projections shape: {task_projections.shape}")
```



## 開發 TODO

* 資料準備
  * 將所有內容寫入 `data` 目錄，並以任務名稱作為子目錄
  * 通用資料格式
    * 在本機以 `HF dataset .jsonl` 儲存，使用 `lm_input` 表示 language model 的輸入，`reference` 為預期輸出。
* 模型推論
  * 傳入模型名稱、checkpoint
  * 執行推論（寫入 `outputs`）／評估
* 資料儲存
  * Kaiser 在某個 save 目錄中做過

* 次要 TODO：目前版本的 VLLM 不相容，需要往回搜尋直到找到可用版本嗎？（推測，尚不確定，但似乎是近期問題） - 之後也許可以修復 VLLM 以加快生成速度...（Millicent）

#
測試
```
python models/evaluate_models.py \
  --model_id LLM360/Crystal \
  --max_new_tokens 5 \
  --chkpt main 

python models/evaluate_models.py \
  --task_name FRCT_CV1_ScrambledWords \
  --max_new_tokens 5 \
  --chkpt main 

```
