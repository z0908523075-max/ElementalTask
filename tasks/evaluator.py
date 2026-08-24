"""任務 evaluator for running 模型 on 任務 with different 後端."""

import json
import torch
import torch.nn.functional as F
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 模型 後端
try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from .base_task import BaseTask


@dataclass
class ModelConfig:
    """設定 for 模型 載入 and 生成."""
    model_id: str
    backend: str  # 'vllm', 'transformers', 'openai', 'together'
    checkpoint: Optional[str] = None
    local_path: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 100
    top_p: float = 1.0
    tensor_parallel_size: Optional[int] = None
    trust_remote_code: bool = True
    quantization: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """設定 for 評估 settings."""
    output_dir: str = "results"
    save_predictions: bool = True
    save_detailed_results: bool = True
    batch_size: int = 1
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # 評估 mode: "exact_match", "continuous", "all"
    eval_mode: str = "exact_match"
    
    # Continuous metrics settings
    compute_loss: bool = True
    compute_perplexity: bool = True


class TaskEvaluator:
    """主要 evaluator 類別 for running 模型 on 任務."""
    
    def __init__(self, model_config: ModelConfig, eval_config: EvaluationConfig):
        self.model_config = model_config
        self.eval_config = eval_config
        self.model = None
        self.tokenizer = None
        self.client = None
        
        # 建立輸出目錄
        Path(self.eval_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """載入model based on the 後端 設定."""
        backend = self.model_config.backend.lower()
        
        if backend == 'vllm':
            self._load_vllm_model()
        elif backend == 'transformers':
            self._load_transformers_model()
        elif backend == 'openai':
            self._load_openai_client()
        elif backend == 'together':
            self._load_together_client()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _load_vllm_model(self):
        """載入model using vLLM 後端."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install it with: pip install vllm")
        
        model_path = self.model_config.local_path or self.model_config.model_id
        tensor_parallel_size = self.model_config.tensor_parallel_size or torch.cuda.device_count()
        
        self.model = vllm.LLM(
            model=model_path,
            tokenizer=model_path,
            revision=self.model_config.checkpoint,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=self.model_config.trust_remote_code,
            quantization=self.model_config.quantization,
        )
    
    def _load_transformers_model(self):
        """載入model using Transformers 後端."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is not available. Please install it with: pip install transformers")
        
        model_path = self.model_config.local_path or self.model_config.model_id
        
        # Try to 載入tokenizer with fallback chain
        tokenizer_loaded = False
        tokenizer_errors = []
        
        # Attempt 1: Try 載入 tokenizer from the 特定 checkpoint
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=self.model_config.checkpoint,
                trust_remote_code=self.model_config.trust_remote_code
            )
            tokenizer_loaded = True
        except Exception as e:
            tokenizer_errors.append(f"Checkpoint tokenizer failed: {e}")
            
            # Attempt 2: Try 載入 from 主要 branch (tokenizers are usually the 相同)
            try:
                print(f"⚠️  Tokenizer loading failed for checkpoint, falling back to 'main' branch...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    revision="main",
                    trust_remote_code=self.model_config.trust_remote_code
                )
                tokenizer_loaded = True
            except Exception as e2:
                tokenizer_errors.append(f"Main branch tokenizer failed: {e2}")
        
        if not tokenizer_loaded:
            raise RuntimeError(f"Failed to load tokenizer after all attempts: {tokenizer_errors}")
        
        # Handle missing pad token (common in GPT-2 based 模型)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"⚠️  No pad token found, using EOS token as pad token: '{self.tokenizer.pad_token}'")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=self.model_config.checkpoint,
            trust_remote_code=self.model_config.trust_remote_code,
            torch_dtype=torch.bfloat16,  # avoid float16 overflow (NaN logits) at early checkpoints
            device_map="auto",
            use_safetensors=False,
        )
    
    def _load_openai_client(self):
        """載入OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not available. Please install it with: pip install openai")
        
        api_key = self.model_config.api_key
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
    
    def _load_together_client(self):
        """載入Together client."""
        if not TOGETHER_AVAILABLE:
            raise ImportError("Together is not available. Please install it with: pip install together")
        
        api_key = self.model_config.api_key
        if not api_key:
            import os
            api_key = os.getenv("TOGETHER_API_KEY")
        
        if not api_key:
            raise ValueError("Together API key not provided")
        
        import os
        os.environ["TOGETHER_API_KEY"] = api_key
        self.client = Together()
    
    def generate(self, prompts: List[str]) -> List[str]:
        """生成responses for a 列表 of 提示."""
        backend = self.model_config.backend.lower()
        
        if backend == 'vllm':
            return self._generate_vllm(prompts)
        elif backend == 'transformers':
            return self._generate_transformers(prompts)
        elif backend in ['openai', 'together']:
            return self._generate_api(prompts)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """生成using vLLM."""
        sampling_params = vllm.SamplingParams(
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            top_p=self.model_config.top_p,
            **self.model_config.generation_kwargs
        )
        
        print(f"Generating {len(prompts)} predictions with vLLM...")
        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _generate_transformers(self, prompts: List[str]) -> List[str]:
        """生成using Transformers."""
        generated_texts = []
        
        # Use tqdm for progress tracking
        for prompt in tqdm(prompts, desc="Generating predictions", unit="prompt"):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Prepare 生成 arguments
            generation_kwargs = {
                "max_new_tokens": self.model_config.max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                **self.model_config.generation_kwargs
            }
            
            # Handle temperature and sampling
            if self.model_config.temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": self.model_config.temperature,
                    "top_p": self.model_config.top_p,
                })
            else:
                generation_kwargs["do_sample"] = False
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode only the 已生成 part (exclude 輸入)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def _generate_api(self, prompts: List[str]) -> List[str]:
        """生成using API 後端 (OpenAI/Together)."""
        generated_texts = []
        
        for prompt in tqdm(prompts, desc="Generating predictions", unit="prompt"):
            for attempt in range(self.eval_config.retry_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_config.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.model_config.temperature,
                        max_tokens=self.model_config.max_tokens,
                        top_p=self.model_config.top_p,
                        **self.model_config.generation_kwargs
                    )
                    generated_texts.append(response.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt == self.eval_config.retry_attempts - 1:
                        print(f"Failed to generate for prompt after {self.eval_config.retry_attempts} attempts: {e}")
                        generated_texts.append("")
                    else:
                        print(f"Attempt {attempt + 1} failed, retrying: {e}")
                        time.sleep(self.eval_config.retry_delay)
        
        return generated_texts
    
    # ============================================================
    # CONTINUOUS METRICS: Loss/Perplexity/Probability Computation
    # ============================================================
    
    def compute_target_metrics(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """
        Compute continuous metrics (loss, perplexity, probability) for each (提示, target) pair.
        
        This provides a much smoother and more informative metric than binary 完全匹配,
        capturing "partial" learning and 模型 confidence.
        
        參數：
            提示: 列表 of 輸入 提示
            targets: 列表 of expected targetoutputs
            
        回傳：
            列表 of dicts with loss, perplexity, probability for each 範例
        """
        backend = self.model_config.backend.lower()
        
        if backend == "transformers":
            return self._compute_target_metrics_transformers(prompts, targets)
        elif backend == "vllm":
            return self._compute_target_metrics_vllm(prompts, targets)
        elif backend in ["openai", "together"]:
            print("⚠️  Continuous metrics have limited support for API backends")
            return self._compute_target_metrics_api(prompts, targets)
        else:
            raise ValueError(f"Continuous metrics not supported for backend: {backend}")
    
    def _compute_target_metrics_transformers(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute loss/perplexity for each (提示, target) pair using Transformers."""
        device = next(self.model.parameters()).device
        results = []
        
        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Computing target metrics"):
            # Tokenize 提示 and full sequence separately
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"]
            full_text = prompt + target
            full_ids = self.tokenizer(full_text, return_tensors="pt", truncation=True)["input_ids"]
            
            prompt_len = prompt_ids.shape[1]
            full_len = full_ids.shape[1]
            
            # If targetadds no 新的 tokens, skip
            if full_len <= prompt_len:
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0
                })
                continue
            
            full_ids = full_ids.to(device)
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # Compute loss only on targettokens
                # Shift: predict token[i+1] from position[i]
                target_logits = logits[0, prompt_len-1:full_len-1, :]  # [target_len, vocab_size]
                target_labels = full_ids[0, prompt_len:full_len]  # [target_len]
                
                num_target_tokens = target_labels.shape[0]
                
                # Compute per-token log probabilities
                log_probs = F.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs[range(num_target_tokens), target_labels]
                
                # Average negative log likelihood (loss)
                avg_loss = -token_log_probs.mean().item()
                
                # Perplexity = exp(loss)
                perplexity = math.exp(avg_loss)
                
                # Probability = exp(sum of log probs)
                total_log_prob = token_log_probs.sum().item()
                probability = math.exp(total_log_prob)
                
                # Normalized probability (geometric mean)
                normalized_prob = math.exp(total_log_prob / num_target_tokens)
            
            results.append({
                "loss": avg_loss,
                "perplexity": perplexity,
                "probability": probability,
                "normalized_probability": normalized_prob,
                "total_log_prob": total_log_prob,
                "num_target_tokens": num_target_tokens
            })
        
        return results
    
    def _compute_target_metrics_vllm(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute loss/perplexity for each (提示, target) pair using vLLM."""
        results = []
        
        # vLLM can 回傳logprobs - we'll use prompt_logprobs
        sampling_params = vllm.SamplingParams(
            max_tokens=1,
            prompt_logprobs=0,  # 回傳logprobs for 提示 tokens
            temperature=0.0
        )
        
        print("Computing target metrics with vLLM...")
        
        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Computing target metrics"):
            full_text = prompt + target
            
            try:
                outputs = self.model.generate([full_text], sampling_params)
                output = outputs[0]
                
                if output.prompt_logprobs is not None:
                    # Find where targetstarts
                    prompt_tokens = self.model.get_tokenizer().encode(prompt)
                    prompt_len = len(prompt_tokens)
                    
                    # 取得logprobs for targettokens
                    target_logprobs = []
                    for i, lp in enumerate(output.prompt_logprobs):
                        if i >= prompt_len and lp is not None:
                            token_id = output.prompt_token_ids[i]
                            if token_id in lp:
                                target_logprobs.append(lp[token_id].logprob)
                    
                    if target_logprobs:
                        total_log_prob = sum(target_logprobs)
                        avg_loss = -sum(target_logprobs) / len(target_logprobs)
                        perplexity = math.exp(avg_loss)
                        probability = math.exp(total_log_prob)
                        normalized_prob = math.exp(total_log_prob / len(target_logprobs))
                        
                        results.append({
                            "loss": avg_loss,
                            "perplexity": perplexity,
                            "probability": probability,
                            "normalized_probability": normalized_prob,
                            "total_log_prob": total_log_prob,
                            "num_target_tokens": len(target_logprobs)
                        })
                        continue
                
                # Fallback if logprobs not 可用
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0
                })
                
            except Exception as e:
                print(f"Warning: Failed to compute metrics: {e}")
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0,
                    "error": str(e)
                })
        
        return results
    
    def _compute_target_metrics_api(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute metrics using API (limited 支援)."""
        # Most chat 模型 don't 支援 logprobs directly
        # 回傳placeholder 結果
        return [
            {
                "loss": None,
                "perplexity": None,
                "probability": None,
                "note": "Continuous metrics not fully supported for API backends"
            }
            for _ in prompts
        ]
    
    def evaluate_task(self, task: BaseTask, split: str = "test") -> Dict[str, Any]:
        """
        評估a 模型 on a 特定 任務.
        
        支援 multiple 評估 modes:
        - "exact_match": Traditional binary 準確率 (預設)
        - "continuous": Loss/perplexity-based 評估 (smoother metrics)
        - "all": Both exact_match and continuous metrics
        """
        eval_mode = self.eval_config.eval_mode
        print(f"Evaluating task: {task.config.name}")
        print(f"Evaluation mode: {eval_mode}")
        
        # 取得task 資料
        task_data = task.get_split(split)
        print(f"Loaded {len(task_data)} examples")
        
        # 建立提示s
        prompts = [task.build_prompt(instance) for instance in task_data]
        
        # 取得targets for continuous metrics
        output_col = task.config.output_column or "answer"
        targets = [str(instance.get(output_col, instance.get("output", ""))) for instance in task_data]
        
        # 初始化results
        results = {
            "task_name": task.config.name,
            "model_id": self.model_config.model_id,
            "backend": self.model_config.backend,
            "checkpoint": self.model_config.checkpoint,
            "split": split,
            "eval_mode": eval_mode,
            "num_examples": len(task_data),
            "metrics": {},
            "config": {
                "model_config": self.model_config.__dict__,
                "eval_config": self.eval_config.__dict__,
                "task_config": task.config.__dict__
            }
        }
        
        predictions = None
        target_metrics = None
        
        # === 完全匹配 評估 ===
        if eval_mode in ["exact_match", "all"]:
            print("\nGenerating predictions for exact match...")
            predictions = self.generate(prompts)
            
            print("Evaluating predictions...")
            exact_match_metrics = task.evaluate(predictions, split=split)
            results["metrics"]["exact_match"] = exact_match_metrics
            results["predictions"] = predictions
        
        # === CONTINUOUS METRICS 評估 ===
        if eval_mode in ["continuous", "all"]:
            print("\nComputing continuous metrics (loss/perplexity)...")
            target_metrics = self.compute_target_metrics(prompts, targets)
            
            # Aggregate metrics
            valid_metrics = [m for m in target_metrics if m.get("loss") is not None and m.get("loss") != float('inf')]
            
            if valid_metrics:
                results["metrics"]["continuous"] = {
                    "mean_loss": sum(m["loss"] for m in valid_metrics) / len(valid_metrics),
                    "mean_perplexity": sum(m["perplexity"] for m in valid_metrics) / len(valid_metrics),
                    "mean_probability": sum(m.get("probability", 0) for m in valid_metrics) / len(valid_metrics),
                    "mean_normalized_probability": sum(m.get("normalized_probability", 0) for m in valid_metrics) / len(valid_metrics),
                    "num_valid_examples": len(valid_metrics),
                    "num_total_examples": len(target_metrics)
                }
            else:
                results["metrics"]["continuous"] = {
                    "error": "No valid metrics computed",
                    "num_valid_examples": 0,
                    "num_total_examples": len(target_metrics)
                }
            
            results["target_metrics"] = target_metrics
        
        # 儲存results
        if self.eval_config.save_predictions or self.eval_config.save_detailed_results:
            self._save_results(results, task_data, prompts, predictions, target_metrics, targets)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], task_data: List[Dict], 
                     prompts: List[str], predictions: Optional[List[str]] = None,
                     target_metrics: Optional[List[Dict]] = None, 
                     targets: Optional[List[str]] = None):
        """儲存evaluation 結果 to 檔案.
        
        For 任務 with subtasks/類別 (like simple_icl), saves separate 檔案
        per 類別 to avoid overwriting.
        """
        model_name = self.model_config.model_id.replace('/', '_')
        task_name = results["task_name"]
        # Sanitize 任務 名稱 for 檔案 路徑 (replace : and , with _)
        task_name_safe = task_name.replace(':', '_').replace(',', '_')
        checkpoint = self.model_config.checkpoint or "main"
        
        base_filename = f"{model_name}_{checkpoint}_{task_name_safe}"
        
        # 儲存summary metrics
        summary_path = Path(self.eval_config.output_dir) / f"{base_filename}_metrics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 儲存detailed 預測 if requested
        if self.eval_config.save_detailed_results:
            # 群組 items by 類別 if present
            category_items = {}  # category_name -> 列表 of items
            
            for i, data in enumerate(task_data):
                item = {
                    "index": i,
                    "input": data.get("input", ""),
                    "ground_truth": data.get("output", ""),
                    "prompt": prompts[i],
                    "metadata": {k: v for k, v in data.items() if k not in ["input", "output"]}
                }
                
                # Add 預測 若可用
                if predictions is not None:
                    item["prediction"] = predictions[i]
                    
                    # Add 正確 field by comparing 預測 to target
                    if targets is not None:
                        pred_clean = predictions[i].split('\n')[0].strip().lower() if predictions[i] else ""
                        target_clean = targets[i].strip().lower() if targets[i] else ""
                        item["correct"] = (pred_clean == target_clean)
                
                # Add targetand continuous metrics 若可用
                if targets is not None:
                    item["target"] = targets[i]
                
                if target_metrics is not None:
                    item["continuous_metrics"] = target_metrics[i]
                
                # 群組 by 類別 if present
                category = data.get("category_name", None)
                if category:
                    if category not in category_items:
                        category_items[category] = []
                    category_items[category].append(item)
                else:
                    # No 類別 - use 預設
                    if "_default" not in category_items:
                        category_items["_default"] = []
                    category_items["_default"].append(item)
            
            # 儲存files - one per 類別 or single 檔案 if no 類別
            if len(category_items) == 1 and "_default" in category_items:
                # No 類別 - 儲存single 檔案
                detailed_path = Path(self.eval_config.output_dir) / f"{base_filename}_detailed.jsonl"
                with open(detailed_path, 'w', encoding='utf-8') as f:
                    for item in category_items["_default"]:
                        f.write(json.dumps(item, default=str) + '\n')
            else:
                # Multiple 類別 - 儲存separate 檔案 per 類別
                for category, items in category_items.items():
                    if category == "_default":
                        continue
                    category_safe = category.replace(':', '_').replace(',', '_').replace(' ', '_')
                    category_filename = f"{model_name}_{checkpoint}_{task_name_safe}_{category_safe}_detailed.jsonl"
                    detailed_path = Path(self.eval_config.output_dir) / category_filename
                    with open(detailed_path, 'w', encoding='utf-8') as f:
                        for item in items:
                            f.write(json.dumps(item, default=str) + '\n')
                    print(f"  Saved {len(items)} items for category '{category}'")
        
        print(f"Results saved to {self.eval_config.output_dir}")


# Convenience function for quick 評估
def evaluate_model_on_task(
    model_config: ModelConfig,
    task: BaseTask,
    eval_config: Optional[EvaluationConfig] = None,
    split: str = "test"
) -> Dict[str, Any]:
    """Convenience function to 評估a 模型 on a 任務."""
    if eval_config is None:
        eval_config = EvaluationConfig()
    
    evaluator = TaskEvaluator(model_config, eval_config)
    return evaluator.evaluate_task(task, split)
