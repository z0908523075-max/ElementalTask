import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import argparse
sys.path.append(os.getcwd())
from datasets import load_dataset
#### OLMO
# 基礎模型路徑

def load_model_revision(model_id: str, ckpt: Optional[str], use_vllm=False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if "LLM360" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=ckpt, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=ckpt, device_map="auto", trust_remote_code=True)
    
    return model, tokenizer

def predict(model, tokenizer, dataset, output_path):
    generated_texts = []
    for example in tqdm(dataset):
        input_text = example["lm_input"]

        # 將輸入 token 化並移到裝置上
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        del inputs["token_type_ids"]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成輸出
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    dataset.add_column("model_output", generated_texts)
    dataset.to_json(output_path, lines=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="在資料集上評估模型。")
    parser.add_argument("--model_id", type=str, required=True, help="來自 Hugging Face 的模型識別碼。")
    parser.add_argument("--chkpt", type=str, help="模型的 checkpoint 識別碼。")
    parser.add_argument("--dataset_path", type=str, required=True, help="要用於評估的資料集路徑。")
    parser.add_argument("--output_path", type=str, required=True, help="儲存評估結果的路徑。")
    parser.add_argument("--subset", type=str, required=True, help="要使用的資料集 split")
    parser.add_argument("--local_dataset", action="store_true", help="使用本機資料集，而不是從 Hugging Face 下載。")
    
    args = parser.parse_args()

    model_id = args.model_id
    # https://huggingface.co/allenai/OLMo-1B-hf/tree/main
    # https://huggingface.co/allenai/OLMo-2-1124-7B/tree/main
    # model_id = "LLM360/CrystalCoder"
    # https://huggingface.co/LLM360/Crystal

    # 載入指定 revision（請替換成實際的 revision 字串，例如 commit hash 或 tag）
    ckpt = args.chkpt  # 這應該與儲存庫中的 revision 名稱一致
    # ckpt = "CrystalCoder_phase1_checkpoint_055500"

    # 在指定 revision 載入模型與 tokenizer
    if "LLM360" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=ckpt)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=ckpt, device_map="auto")

    dataset = load_dataset("your_dataset_name", split="train")

    # 生成示範輸出
    prompt = "The universe began with"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
