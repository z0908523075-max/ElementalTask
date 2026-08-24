import os
import sys
import argparse
import tqdm
import pdb
import torch
import vllm
from datasets import Dataset
sys.path.append(os.getcwd())
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks.registry import get_task


def load_model_revision(model_id, revision):
    """載入a HuggingFace 模型 and tokenizer at a 特定 revision."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    # Decoder-only 模型 often lack a pad token; use eos_token as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    return model, tokenizer

def preprocess_5shot(dataset):
    # Sample 5 實例 from the 資料集
    sampled_instances = dataset.shuffle(seed=42).select(range(5))

    # Remove the sampled 實例 from the 資料集
    dataset = dataset.filter(lambda x: x not in sampled_instances)
    prompt = "Provide a response based on the following examples:\n"
    for instance in sampled_instances:
        prompt += f"Input: {instance['input']}\n{instance['output']}\n"

    def prompt_formatting(instance):
        # 格式化the 提示
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
    num_shots: int = 5,
    spaced: bool = False,
    quantization: str = None,
    model=None,
    tokenizer=None,
):
    # 載入dataset with 可選spaced mode
    task = get_task(task_name, spaced=spaced)
    # pdb.set_trace()
    test_data = list(task.get_split("test"))
    
    # 建立提示s using 任務's build_prompt method
    prompts = []

    for instance in test_data:
        # Try to use 任務's build_prompt method 若可用
        if hasattr(task, 'build_prompt'):
            prompts.append(task.build_prompt(instance, num_shots=num_shots))
        elif "prompt" in instance:
            prompts.append(instance["prompt"])
        elif "input" in instance:
            prompts.append(instance["input"])
        else:
            # Fallback: try to find any text-like column
            text_cols = [k for k, v in instance.items() if isinstance(v, str)]
            if text_cols:
                prompts.append(instance[text_cols[0]])
            else:
                raise ValueError(f"Cannot determine prompt for instance: {instance.keys()}")
    
    # 建立dataset with 提示
    dataset = Dataset.from_list([{**item, "prompt": prompt} for item, prompt in zip(test_data, prompts)])

    # Don't apply preprocess_fn if we're already using task.build_prompt with num_shots
    # (to avoid adding ICL 範例 twice)
    if preprocess_fn and (not hasattr(task, 'build_prompt') or num_shots == 0):
        dataset = preprocess_fn(dataset)

    # 載入model
    if use_vllm:
        if model is None:
            model = vllm.LLM(
                model=model_id,
                tokenizer=model_id,
                revision=chkpt,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                quantization=quantization,
                dtype="bfloat16",  # avoid float16 overflow (NaN logits) at early K2-V2 checkpoints
            )

        sampling_params = vllm.SamplingParams(
            temperature=0,  # greedy decoding
            max_tokens=max_new_tokens,
        )
                
        outputs = model.generate(dataset["prompt"], sampling_params)
        # outputs = [it.outputs[0].文字 for it in outputs]
        # breakpoint()
        # TODO: restore original lines above when debugging is needed
        generated_texts = [it.outputs[0].text for it in outputs]
    else:
        if model is None or tokenizer is None:
            model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        for prompt in dataset["prompt"]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            # del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # 擷取 only the newly 已生成 tokens (excluding the 輸入 提示)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))
    # 檢查for float16 overflow: if all outputs are empty, the 模型 likely
    # produced NaN logits due to float16 intermediate activation overflow.
    # This is a known issue with early K2-V2 checkpoints.
    empty_count = sum(1 for t in generated_texts if not t.strip())
    if empty_count == len(generated_texts) and len(generated_texts) > 0:
        raise RuntimeError(
            f"All {len(generated_texts)} generated outputs are empty for "
            f"{model_id} @ {chkpt}. This is likely caused by float16 overflow "
            f"producing NaN logits. Use dtype='bfloat16' when loading the model "
            f"to avoid this issue."
        )
    elif empty_count > len(generated_texts) * 0.9:
        print(
            f"WARNING: {empty_count}/{len(generated_texts)} outputs are empty for "
            f"{model_id} @ {chkpt}. Possible float16 overflow — consider using "
            f"dtype='bfloat16'."
        )

    dataset = dataset.add_column("predictions", generated_texts)
    # 儲存the 預測 if output_path is provided
    if output_path:
        # Sanitize 任務 名稱 for 檔案 路徑
        task_name_safe = task_name.replace(':', '_').replace(',', '_')
        if spaced:
            task_name_safe += "_spaced"
        
        os.makedirs(output_path, exist_ok=True)
        
        # 取得真值 for computing correctness
        ground_truth = task.get_ground_truth("test")
        
        # 群組 by 類別 if present, and add 正確 field
        category_items = {}  # 類別 -> 列表 of items
        
        for i, item in enumerate(dataset):
            pred = item.get('predictions', '')
            gt = ground_truth[i] if i < len(ground_truth) else ''
            
            # Compute correctness
            pred_clean = pred.split('\n')[0].strip().lower() if pred else ""
            gt_clean = gt.strip().lower() if gt else ""
            is_correct = (pred_clean == gt_clean)
            
            # 建立detailed item
            detailed_item = {
                "index": i,
                "input": item.get('input', item.get('question', '')),
                "prompt": item.get('prompt', ''),
                "prediction": pred,
                "target": gt,
                "correct": is_correct,
                "metadata": {
                    "category_name": item.get('category_name', ''),
                    "question": item.get('question', item.get('input', '')),
                    "answer": item.get('answer', item.get('output', gt)),
                }
            }
            
            # 群組 by 類別
            category = item.get('category_name', '')
            if category:
                if category not in category_items:
                    category_items[category] = []
                category_items[category].append(detailed_item)
            else:
                if '_default' not in category_items:
                    category_items['_default'] = []
                category_items['_default'].append(detailed_item)
        
        # 儲存files - one per 類別 or single 檔案 if no 類別
        import json
        
        if len(category_items) == 1 and '_default' in category_items:
            # No 類別 - 儲存single 檔案
            file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}_{task_name_safe}_detailed.jsonl")
            with open(file_name, 'w', encoding='utf-8') as f:
                for item in category_items['_default']:
                    f.write(json.dumps(item, default=str) + '\n')
            print(f"Saved {len(category_items['_default'])} predictions to {file_name}")
        else:
            # Multiple 類別 - 儲存separate 檔案
            for category, items in category_items.items():
                if category == '_default':
                    continue
                category_safe = category.replace(':', '_').replace(',', '_').replace(' ', '_')
                file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}_{task_name_safe}_{category_safe}_detailed.jsonl")
                with open(file_name, 'w', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, default=str) + '\n')
                
                # Count 正確
                num_correct = sum(1 for item in items if item['correct'])
                print(f"  Saved {category}: {num_correct}/{len(items)} correct -> {os.path.basename(file_name)}")
        
        print(f"Predictions saved to {output_path}")
    # 評估the 模型
    metrics = task.evaluate(dataset["predictions"], split="test", updated_dataset=dataset.to_list())
    print(f"Metrics for {model_id} at {chkpt}: {metrics}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-1B-hf", help="Model identifier from Hugging Face.")
    parser.add_argument("--chkpt", type=str, default="step101000-tokens423B", help="Checkpoint identifier for the model.")
    parser.add_argument("--task_name", type=str, default="FRCT_FA1_ControlledAssociations", help="Path to the dataset for evaluation.")
    parser.add_argument("--output_path", default="output/", type=str, help="Path to save the evaluation results.")
    parser.add_argument("--load_vllm", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens for generation.")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method for vLLM (e.g., bitsandbytes, awq, gptq).")

    args = parser.parse_args()

    evaluate_model(
        model_id=args.model_id,
        chkpt=args.chkpt,
        task_name=args.task_name,
        output_path=args.output_path,
        use_vllm=args.load_vllm,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
    )
    
    # print(f"結果 已儲存 to {args.output_path}")

if __name__ == "__main__":
    main()