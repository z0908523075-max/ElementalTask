* 用於分析的模型（需要 ckpt）

    - Olmo 1
    - Olmo 2
    - LLM360
    - Emmy 的臨時 ckpts

* 任務（第一階段 - 健全性檢查）

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

* 複雜任務：我們如何用提出的元素任務構造複雜任務？
    - extractiveQA = 理解 + 推理 + 精確複製
    - Open-domain QA = 記憶 + 理解 + 推理 + 語言學
    - Natural Language Inference = 理解 + 推理


## 開發 TODO

* 資料準備
  * 將所有內容寫入 `data` 目錄，並以任務名稱作為子目錄
  * 通用資料格式
    * 在本機儲存為 `HF dataset .jsonl`，其中 `lm_input` 表示語言模型的輸入，`reference` 為預期輸出。
* 模型推論
  * 傳入模型名稱與 checkpoint
  * 執行推論（寫入 `outputs`）／評估
* 資料儲存
  * Kaiser 曾在某個儲存目錄中處理過

* 次要 TODO：目前版本的 VLLM 不相容，是否需要往回找出可用版本？（推測，尚不確定，但看起來像是近期問題）— 之後也許可以修復 VLLM 以加快生成速度……（Millicent）

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
