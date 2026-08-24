import os
import sys
import argparse
import tqdm
import torch
import vllm
sys.path.append(os.getcwd())
from datasets import load_dataset, load_from_disk
from scripts.inference import load_model_revision

def evaluate_model(
    model_id: str,
    chkpt: str,
    dataset_path: str,
    output_path: str,
    local_dataset: bool = False,
    subset: str = "test",
    use_vllm: bool = True,
    max_new_tokens: int = 100,
):    
    if local_dataset:
        # 載入本機資料集
        data = load_from_disk(dataset_path)
    else:
        data = load_dataset(dataset_path, subset, split="validation")
    
    # TODO: 這是來自 kilt 的格式，之後可能需要泛化
    prompts = [item["input"] for item in data]
    answers = [item["output"][0]["answer"] for item in data]
    if use_vllm:
        model = vllm.LLM(
            model=model_id,
            tokenizer=model_id,
            revision=chkpt,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=0,  # 貪婪解碼
            max_tokens=max_new_tokens,
        )
                
        outputs = model.generate(prompts, sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        for prompt in prompts[:10]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成輸出
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(generated_texts)

        breakpoint()
        raise NotImplementedError

    acc = sum(
        1 if output.strip() == answer.strip() else 0 for output, answer in zip(outputs, answers)
    ) / len(answers)
    print(f"Accuracy: {acc:.4f}")



def main():
    parser = argparse.ArgumentParser(description="在資料集上評估模型。")
    parser.add_argument("--model_id", type=str, required=True, help="來自 Hugging Face 的模型識別碼。")
    parser.add_argument("--chkpt", type=str, help="模型的 checkpoint 識別碼。")
    parser.add_argument("--dataset_path", type=str, required=True, help="要用於評估的資料集路徑。")
    parser.add_argument("--output_path", type=str, required=True, help="儲存評估結果的路徑。")
    parser.add_argument("--subset", type=str, required=True, help="要使用的資料集 split。")
    parser.add_argument("--local_dataset", action="store_true", help="使用本機資料集，而不是從 Hugging Face 下載。")
    parser.add_argument("--load_vllm", action="store_true", help="若使用 vllm 進行推論則設為 True，否則為 False；預設為 False。")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成時的最大 token 數。")
    
    args = parser.parse_args()

    # 評估邏輯入口
    print(f"正在資料集 {args.dataset_path} 上評估模型 {args.model_id}...")

    # VLLM 預設停用多程序，這裡需要顯式啟用
    if args.load_vllm:
        os.environ["LLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    evaluate_model(
        args.model_id,
        args.chkpt,
        args.dataset_path,
        args.output_path,
        args.local_dataset,
        args.subset,
        args.load_vllm,
        args.max_new_tokens,
    )
    
    print(f"結果已儲存至 {args.output_path}")

if __name__ == "__main__":
    main()