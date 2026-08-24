# OLMo-2 模型 Checkpoints

## 1B 與 7B 模型之間的關鍵差異

### 模型名稱
- **1B Model:** `allenai/OLMo-2-0425-1B`（於 2025 年 4 月發布）
- **7B Model:** `allenai/OLMo-2-1124-7B`（於 2024 年 11 月發布）

### 訓練 Token 縮放
這兩個模型具有不同的每步 token 比率：
- **1B:** 每 1000 steps 約 ~2.1B tokens
- **7B:** 每 1000 steps 約 ~4.2B tokens（剛好 2x）

### Checkpoint 比較

| Step | 1B Tokens | 7B Tokens | 備註 |
|------|-----------|-----------|-------|
| 10,000 | 21B | 42B | 早期 checkpoint |
| 100,000 | 210B | 419B | 約多 100x steps |
| 200,000 | 420B | 839B | 2x tokens (1B) vs 2x tokens (7B) |
| 300,000 | 630B | 1,259B | 3x 基準 tokens |
| 400,000 | 839B | 1,678B | 4x 基準 tokens |
| 500,000 | 1,049B | 2,098B | main 之前的最終 checkpoint |
| main | ~1.1T | ~2.1T | 最終訓練完成模型 |

## 可用設定

### 1. `olmo2_checkpoints_1b.json`
僅包含 1B 模型，共 12 個 checkpoints（10k、50k、100k、...、500k、main）

### 2. `olmo2_checkpoints.json`
僅包含 7B 模型，共 12 個 checkpoints（10k、50k、100k、...、500k、main）

### 3. `olmo2_1b_7b_checkpoints.json`
同時包含兩個模型，各有 7 個 checkpoints（10k、100k、200k、300k、400k、500k、main）
- 更快地進行跨模型大小比較的評估
- 每 100k steps 選取一個 checkpoint

## 注意事項

- 所有 checkpoints 都使用 `stage1-stepXXXXX-tokensYYYB` 格式
- `main` branch 永遠位於最後，並代表最終訓練完成模型
- 7B checkpoints 每個約 ~14GB，1B checkpoints 每個約 ~2GB
- 請務必將 `HF_HOME` 設定到有足夠空間的目錄

---

# LLM360 K2-V2 Checkpoints

## 模型概覽
- **Model:** `LLM360/K2-V2` ([HuggingFace](https://huggingface.co/LLM360/K2-V2))
- **Parameters:** 70B
- **Architecture:** 僅 decoder 的 transformer，具 grouped-query attention、RMSNorm、80 層
- **Vocab Size:** 250,000
- **Pre-training Tokens:** 約 ~12T tokens
- **Pre-training Sequence Length:** 8,192
- **License:** Apache 2.0

## Token 計算
- 全域 token batch size：**B = 9.8 × 10^6 tokens/step**（1200 sequences × 8192 seq_len）
- 總訓練步數：**T = 1.25 × 10^6**
- 總 tokens：**D = 12.25T**
- 公式：`tokens = step_number × 9.8M`

## Checkpoint 格式
Pretrain checkpoints 以 branches/tags 的形式儲存在 HuggingFace repo 中，格式為 `base_XXXXXXX`（step number，補零至 7 位數）。最終 pretrain checkpoint 為 `base_final`。

所有 checkpoints：https://huggingface.co/LLM360/K2-V2/tree/base_final

## 可用設定

### `k2v2_checkpoints.json`
在整個訓練過程中均勻取樣的 11 個 checkpoints，從早期（step 20k）到最終：

| Checkpoint | Step |
|-----------|------|
| `base_0020000` | 20,000 |
| `base_0125000` | 125,000 |
| `base_0265000` | 265,000 |
| `base_0405000` | 405,000 |
| `base_0545000` | 545,000 |
| `base_0685000` | 685,000 |
| `base_0825000` | 825,000 |
| `base_0965000` | 965,000 |
| `base_1105000` | 1,105,000 |
| `base_1245000` | 1,245,000 |
| `base_final` | 最終 |

## 載入
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("LLM360/K2-V2", revision="base_0720000", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("LLM360/K2-V2")
```

## 注意事項
- 70B checkpoints 非常大（每個約 ~140GB）；請確保磁碟空間足夠，並適當設定 `HF_HOME`
- tokenizer 在所有 checkpoints 間皆相同；從 `main` 載入即可
- 這是 base（pretrained）模型，而非 instruction-tuned。若需 instruction-tuned 版本，請參見 `LLM360/K2-V2-Instruct`
- K2-V2 另有 mid-training checkpoints（`mid_1_*`、`mid_2_*`、`mid_3_*`、`mid_4_*`），但未包含於預設設定中

---

# LLM360 CrystalCoder Checkpoints

## 模型概覽
- **Model:** `LLM360/CrystalCoder` ([HuggingFace](https://huggingface.co/LLM360/CrystalCoder))
- **Parameters:** 7B
- **Architecture:** 類 GPT（等同 LLaMA-7B），具 Maximal Update Parameterization (muP)、LayerNorm，以及套用於前 25% hidden dims 的 Rotary position embeddings
- **Vocab Size:** 32,032
- **Pre-training Tokens:** 橫跨 3 個 phases 的約 ~1.4T tokens
- **Pre-training Sequence Length:** 2,048
- **License:** Apache 2.0

## 訓練階段
CrystalCoder 使用不同資料混合方式分 3 個 phases 訓練：

| Phase | Data | Tokens | Steps | Cumulative Tokens |
|-------|------|--------|-------|-------------------|
| 1 | SlimPajama（前半） | 345B | 79,721 | 345B |
| 2 | SlimPajama（後半）+ StarCoder (2x) | 927B | 214,387 | 1,272B |
| 3 | Python/web data + SlimPajama sample | 110B | 27,728 | 1,382B |

每步 tokens：~4.3M（phase 1-2）、~4.0M（phase 3）

## Checkpoint 格式
Checkpoints 以 branches 形式儲存在 HuggingFace repo 中，格式為 `CrystalCoder_phase{N}_checkpoint_{XXXXXX}`（step number，補零至 6 位數）。最終 checkpoint 為 `CrystalCoder_phase3_checkpoint_027728`（也可作為 `main` 取得）。

所有可用 checkpoints 總數：約 120 個，涵蓋全部 3 個 phases。

## 可用設定

### `crystal_checkpoints.json`
在整個訓練過程中均勻取樣的 11 個 checkpoints（按 token 數計）：

| Checkpoint | Phase | Cumulative Tokens |
|-----------|-------|-------------------|
| `CrystalCoder_phase1_checkpoint_001500` | 1 | ~6.5B |
| `CrystalCoder_phase1_checkpoint_033000` | 1 | ~143B |
| `CrystalCoder_phase1_checkpoint_064500` | 1 | ~279B |
| `CrystalCoder_phase2_checkpoint_018000` | 2 | ~423B |
| `CrystalCoder_phase2_checkpoint_051000` | 2 | ~565B |
| `CrystalCoder_phase2_checkpoint_081000` | 2 | ~695B |
| `CrystalCoder_phase2_checkpoint_114000` | 2 | ~838B |
| `CrystalCoder_phase2_checkpoint_144000` | 2 | ~967B |
| `CrystalCoder_phase2_checkpoint_174000` | 2 | ~1,097B |
| `CrystalCoder_phase2_checkpoint_207000` | 2 | ~1,239B |
| `CrystalCoder_phase3_checkpoint_027728` | 3 | ~1,382B |

### `crystal_sanity_check.json`
單一最終 checkpoint，用於快速測試。

## 載入
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LLM360/CrystalCoder",
    revision="CrystalCoder_phase1_checkpoint_055500",
    trust_remote_code=True  # 自訂 muP 架構必需
)
tokenizer = AutoTokenizer.from_pretrained("LLM360/CrystalCoder", trust_remote_code=True)
```

## 注意事項
- **`trust_remote_code=True` 為必填** — CrystalCoder 使用帶有 muP 修改的自訂架構
- 7B checkpoints 每個約 ~13GB（3 個 shards）；請確保磁碟空間足夠，並適當設定 `HF_HOME`
- tokenizer 在所有 checkpoints 間皆相同
- HuggingFace 上總共有 250 個 branches（包含較舊的 `mdl_phase*_step_*` 命名慣例 — 請使用 `CrystalCoder_phase*_checkpoint_*` 命名）
