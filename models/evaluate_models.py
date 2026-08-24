import os
import sys
import argparse
import tqdm
import pdb
import torch
import vllm
from datasets import Dataset
sys.path.append(os.getcwd())
from scripts.inference import load_model_revision
from tasks.base_task import get_task

def preprocess_5shot(dataset):
    # 從資料集中抽樣 5 個實例
    sampled_instances = dataset.shuffle(seed=42).select(range(5))

    # 從資料集中移除已抽樣的實例
    dataset = dataset.filter(lambda x: x not in sampled_instances)
    prompt = "Provide a response based on the following examples:\n"
    for instance in sampled_instances:
        prompt += f"Input: {instance['input']}\n{instance['output']}\n"

    def prompt_formatting(instance):
        # 格式化提示
        instance["prompt"] = prompt + f"Input: {instance['input']}\n"
        return instance
    dataset = dataset.map(prompt_formatting)
    return dataset

def evaluate_model(
    model_id: str,
    chkpt: str,
    task_name: str,
    output_path: str = None,
    use_vllm: bool = True,
    max_new_tokens: int = 100,
    preprocess_fn: callable = preprocess_5shot,
):
    # 載入資料集
    task = get_task(task_name)
    # pdb.set_trace()
    dataset = Dataset.from_list(list(task.get_split("test")))

    if preprocess_fn:
        dataset = preprocess_fn(dataset)

    # 載入模型
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
                
        outputs = model.generate(dataset["prompt"] if "prompt" in dataset else dataset["input"], sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        prompts = dataset["prompt"] if "prompt" in dataset else dataset["input"]
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            # del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成輸出
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    dataset = dataset.add_column("predictions", generated_texts)
    # 如果提供 output_path，則儲存預測結果
    if output_path:
        file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}_{task_name}.jsonl")
        os.makedirs(output_path, exist_ok=True)
        dataset.to_json(file_name, orient="records", lines=True)
        print(f"預測結果已儲存至 {output_path}")
    # 評估模型
    metrics = task.evaluate(dataset["predictions"], split="test", updated_dataset=dataset.to_list())
    print(f"{model_id} 在 {chkpt} 的指標：{metrics}")

def main():
    parser = argparse.ArgumentParser(description="在資料集上評估模型。")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-1B-hf", help="來自 Hugging Face 的模型識別碼。")
    parser.add_argument("--chkpt", type=str, default="step101000-tokens423B", help="模型的 checkpoint 識別碼。")
    parser.add_argument("--task_name", type=str, default="FRCT_FA1_ControlledAssociations", help="要評估的任務名稱。")
    parser.add_argument("--output_path", default="output/", type=str, help="儲存評估結果的路徑。")
    parser.add_argument("--load_vllm", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成時的最大 token 數。")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_id=args.model_id,
        chkpt=args.chkpt,
        task_name=args.task_name,
        output_path=args.output_path,
        use_vllm=args.load_vllm,
        max_new_tokens=args.max_new_tokens
    )
    
    # print(f"結果已儲存至 {args.output_path}")

if __name__ == "__main__":
    main()