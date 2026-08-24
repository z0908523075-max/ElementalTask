#!/usr/bin/env python3
"""評估a 模型 across multiple checkpoints on all discovered 任務.

This script runs 評估 across different 模型 checkpoints, useful for
tracking performance improvements during training.

用法：
    # 評估OLMo-2-1124-7B across multiple checkpoints
    python scripts/eval_across_checkpoints.py \
        --model_id allenai/OLMo-2-1124-7B \
        --checkpoints step10000-tokens42B step50000-tokens210B 主要 \
        --output_path 結果/olmo2_7b_progression \
        --任務 basic_arithmetic copying simple_icl

    # Or from a json 設定 檔案
    python scripts/eval_across_checkpoints.py \
        --model_configs configs/model_checkpoints.json \
        --output_path 結果/multi_model_eval
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


from tasks.registry import TaskRegistry


def sanitize_task_name(task_name: str, spaced: bool = False) -> str:
    """Sanitize 任務 名稱 for use in 檔案 路徑 (replace : with _)."""
    sanitized = task_name.replace(':', '_').replace(',', '_')
    if spaced:
        sanitized += "_spaced"
    return sanitized


def check_existing_results(output_dir: Path, model_id: str, checkpoint: str, task_name: str):
    """檢查if 結果 already exist and 載入them 若可用.
    
    Looks for detailed JSONL 檔案 with per-category 結果.
    """
    model_name = model_id.replace('/', '_')
    chkpt_name = checkpoint.replace('/', '_')
    task_name_safe = sanitize_task_name(task_name)
    
    # 檢查for 新的 detailed 預測 檔案 (per-category or single)
    # Pattern: {model}_{checkpoint}_{task}_detailed.jsonl or {model}_{checkpoint}_{task}_{category}_detailed.jsonl
    detailed_pattern = f"{model_name}_{chkpt_name}_{task_name_safe}*_detailed.jsonl"
    detailed_files = list(output_dir.glob(detailed_pattern))
    
    # Also 檢查for 舊的 格式化for backwards compatibility
    old_predictions_file = output_dir / f"{model_name}_{chkpt_name}_{task_name_safe}.jsonl"
    
    if not detailed_files and (not old_predictions_file.exists() or old_predictions_file.stat().st_size == 0):
        return None
    
    try:
        import json
        predictions = []
        num_correct = 0
        
        if detailed_files:
            # 載入from 新的 detailed 格式
            for detailed_file in detailed_files:
                with open(detailed_file, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        predictions.append(item)
                        if item.get('correct', False):
                            num_correct += 1
            
            print(f"  ✓ Found cached results: {num_correct}/{len(predictions)} correct ({len(detailed_files)} files)")
        else:
            # 載入from 舊的 格式
            with open(old_predictions_file, 'r') as f:
                for line in f:
                    predictions.append(json.loads(line))
            print(f"  ✓ Found cached results with {len(predictions)} predictions (old format)")
        
        if len(predictions) == 0:
            return None
        
        # Reconstruct metrics from 預測
        metrics = {
            "cached": True,
            "predictions_file": str(detailed_files[0] if detailed_files else old_predictions_file),
            "num_predictions": len(predictions)
        }
        
        # Add 準確率 if we have 正確 field
        if detailed_files and len(predictions) > 0:
            metrics["accuracy"] = num_correct / len(predictions)
            metrics["correct"] = num_correct
            metrics["total"] = len(predictions)
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠️  Error loading cached results: {e}")
        return None


def run_single_evaluation(
    model_id: str,
    checkpoint: str,
    tasks: List[str],
    output_base: Path,
    max_new_tokens: int,
    load_vllm: bool,
    num_shots: int,
    skip_existing: bool = True,
    eval_mode: str = "exact_match",
    spaced: bool = False,
    quantization: str = None,
) -> Dict[str, Any]:
    """Run 評估 for a single 模型 checkpoint."""
    
    # 建立checkpoint-特定 輸出 目錄
    chkpt_name = checkpoint.replace('/', '_')
    output_dir = output_base / f"{model_id.split('/')[-1]}_{chkpt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_id} @ {checkpoint}")
    print(f"Eval mode: {eval_mode}")
    print(f"Spaced mode: {spaced}")
    print(f"{'='*70}")
    
    # 初始化evaluator for continuous metrics (reuse across 任務)
    evaluator = None
    if eval_mode in ["continuous", "all"]:
        from tasks.evaluator import TaskEvaluator, ModelConfig, EvaluationConfig

        backend = "vllm" if load_vllm else "transformers"
        model_config = ModelConfig(
            model_id=model_id,
            backend=backend,
            checkpoint=checkpoint,
            max_tokens=max_new_tokens,
            quantization=quantization,
        )
        eval_config = EvaluationConfig(
            output_dir=str(output_dir),
            eval_mode=eval_mode,
            save_predictions=True,
            save_detailed_results=True,
        )
        evaluator = TaskEvaluator(model_config, eval_config)

    # 載入model once and reuse across all 任務 for this checkpoint
    # (avoids repeated init/teardown overhead)
    vllm_model = None
    hf_model = None
    hf_tokenizer = None
    if eval_mode == "exact_match" and load_vllm:
        import vllm as _vllm
        import torch
        vllm_model = _vllm.LLM(
            model=model_id,
            tokenizer=model_id,
            revision=checkpoint,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            quantization=quantization,
            gpu_memory_utilization=0.9,
            max_model_len=1024,  # 提示 are short; avoids KV cache OOM on large 模型
            dtype="bfloat16",  # avoid float16 overflow (NaN logits) at early K2-V2 checkpoints
        )
    elif eval_mode == "exact_match" and not load_vllm:
        from models.evaluate_models import load_model_revision
        print(f"  Loading HuggingFace model: {model_id} @ {checkpoint}")
        hf_model, hf_tokenizer = load_model_revision(model_id, checkpoint)

    results = []
    for i, task_name in enumerate(tasks, 1):
        display_name = f"{task_name} (spaced)" if spaced else task_name
        print(f"\n[{i}/{len(tasks)}] Task: {display_name}")

        # Use sanitized 任務 名稱 for 檔案 操作
        task_name_sanitized = sanitize_task_name(task_name, spaced=spaced)

        # 檢查for existing 結果
        if skip_existing:
            existing_metrics = check_existing_results(output_dir, model_id, checkpoint, task_name_sanitized)
            if existing_metrics is not None:
                results.append({
                    "model": model_id,
                    "checkpoint": checkpoint,
                    "task": task_name_sanitized,
                    "success": True,
                    "metrics": existing_metrics,
                    "error": "",
                    "cached": True,
                })
                continue

        try:
            if eval_mode == "exact_match":
                # Use original evaluate_model for 完全匹配 only
                from models.evaluate_models import evaluate_model
                metrics = evaluate_model(
                    model_id=model_id,
                    chkpt=checkpoint,
                    task_name=task_name,
                    output_path=str(output_dir),
                    use_vllm=load_vllm,
                    max_new_tokens=max_new_tokens,
                    preprocess_fn=None,
                    num_shots=num_shots,
                    spaced=spaced,
                    quantization=quantization,
                    model=vllm_model or hf_model,
                    tokenizer=hf_tokenizer,
                )
            else:
                # Use 新的 evaluator for continuous/all modes
                from tasks.registry import get_task
                task = get_task(task_name, spaced=spaced)
                eval_result = evaluator.evaluate_task(task, split="test")
                
                # Flatten metrics for CSV compatibility
                metrics = {}
                if "exact_match" in eval_result.get("metrics", {}):
                    metrics.update(eval_result["metrics"]["exact_match"])
                if "continuous" in eval_result.get("metrics", {}):
                    cont = eval_result["metrics"]["continuous"]
                    metrics["mean_loss"] = cont.get("mean_loss")
                    metrics["mean_perplexity"] = cont.get("mean_perplexity")
                    metrics["mean_probability"] = cont.get("mean_probability")
                    metrics["mean_normalized_probability"] = cont.get("mean_normalized_probability")
            
            results.append({
                "model": model_id,
                "checkpoint": checkpoint,
                "task": task_name_sanitized,
                "success": True,
                "metrics": metrics,
                "error": "",
                "cached": False,
            })
            
        except Exception as e:
            print(f"❌ Failed to evaluate {display_name}: {e}")
            results.append({
                "model": model_id,
                "checkpoint": checkpoint,
                "task": task_name_sanitized,
                "success": False,
                "metrics": {},
                "error": str(e),
                "cached": False,
            })
    
    # CRITICAL CLEANUP: Destroy vLLM engine to free GPU 記憶 and tear down
    # distributed 處理 群組 before the 下一個 checkpoint iteration.
    # Without this, gloo/nccl can fail to rebind on subsequent inits.
    if vllm_model is not None:
        import gc
        import contextlib
        del vllm_model
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception as e:
            print(f"  ⚠️  destroy_model_parallel failed: {e}")
        try:
            import torch.distributed
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as e:
            print(f"  ⚠️  destroy_process_group failed: {e}")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("  ✓ vLLM engine cleaned up")

    if hf_model is not None:
        import gc
        del hf_model
        del hf_tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("  ✓ HuggingFace model cleaned up")

    if evaluator is not None:
        import gc
        del evaluator
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("  ✓ Evaluator cleaned up")

    return {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models across multiple checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 模型 specification (either individual or 設定 檔案)
    parser.add_argument("--model_id", type=str,
                        help="Single model identifier (HuggingFace model ID)")
    parser.add_argument("--checkpoints", nargs="+",
                        help="List of checkpoints/revisions to evaluate")
    parser.add_argument("--model_configs", type=str,
                        help="JSON file with model->checkpoints mapping")
    
    # 評估 settings
    parser.add_argument("--output_path", type=str, default="results/checkpoint_eval",
                        help="Base directory to save results")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all)")
    parser.add_argument("--skip_tasks", nargs="+", default=None,
                        help="Tasks to skip")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--load_vllm", action="store_true",
                        help="Use vLLM for faster inference")
    parser.add_argument("--num_shots", type=int, default=5,
                        help="Number of in-context learning examples")
    parser.add_argument("--force_reeval", action="store_true",
                        help="Force re-evaluation even if results exist")
    parser.add_argument("--eval_mode", type=str, default="exact_match",
                        choices=["exact_match", "continuous", "all"],
                        help="Evaluation mode: exact_match (default), continuous (loss/perplexity), or all")
    parser.add_argument("--spaced", action="store_true",
                        help="Use spaced version of tasks (add spaces between characters for reversal/letter tasks)")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method for vLLM (e.g., bitsandbytes, awq, gptq). Useful for large models.")

    args = parser.parse_args()
    
    # 驗證 輸入
    if args.model_configs:
        with open(args.model_configs, 'r') as f:
            model_configs = json.load(f)
    elif args.model_id and args.checkpoints:
        model_configs = {args.model_id: args.checkpoints}
    else:
        parser.error("Must provide either --model_configs or both --model_id and --checkpoints")
    
    # 建立輸出目錄
    output_base = Path(args.output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 儲存run 設定
    run_config = {
        "model_configs": model_configs,
        "tasks": args.tasks,
        "skip_tasks": args.skip_tasks,
        "max_new_tokens": args.max_new_tokens,
        "load_vllm": args.load_vllm,
        "num_shots": args.num_shots,
        "eval_mode": args.eval_mode,
        "spaced": args.spaced,
        "quantization": args.quantization,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_base / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    print("\n" + "="*70)
    print("CHECKPOINT EVALUATION CONFIGURATION")
    print("="*70)
    print(f"  Models: {list(model_configs.keys())}")
    print(f"  Total checkpoints: {sum(len(chkpts) for chkpts in model_configs.values())}")
    print(f"  num_shots: {args.num_shots}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  load_vllm: {args.load_vllm}")
    print(f"  eval_mode: {args.eval_mode}")
    print(f"  spaced: {args.spaced}")
    print(f"  quantization: {args.quantization}")
    
    # Discover 任務
    print("\n" + "="*70)
    print("DISCOVERING TASKS")
    print("="*70)
    
    registry = TaskRegistry()
    all_tasks = registry.discover_tasks()
    
    # 篩選 任務. 支援 subtask syntax '任務:sub1,sub2' by matching 基礎 任務 名稱.
    def _base_name(spec: str) -> str:
        return spec.split(':', 1)[0] if isinstance(spec, str) and ':' in spec else spec

    if args.tasks:
        task_names = []
        missing = []
        for spec in args.tasks:
            base = _base_name(spec)
            if base in all_tasks:
                task_names.append(spec)
            else:
                missing.append(spec)
        if missing:
            print(f"\n⚠️  Tasks not found: {set(missing)}")
    else:
        task_names = list(all_tasks.keys())
    
    if args.skip_tasks:
        # Allow skip entries with subtask syntax as well
        skip_bases = {_base_name(s) for s in args.skip_tasks}
        task_names = [t for t in task_names if _base_name(t) not in skip_bases]
        print(f"\nSkipping tasks: {args.skip_tasks}")
    
    print(f"\nTasks to evaluate ({len(task_names)}): {task_names}")
    
    # Run evaluations across all model-checkpoint combinations
    print("\n" + "="*70)
    print("RUNNING EVALUATIONS")
    print("="*70)
    
    all_results = []
    total_evals = sum(len(chkpts) for chkpts in model_configs.values())
    eval_count = 0
    
    for model_id, checkpoints in model_configs.items():
        for checkpoint in checkpoints:
            eval_count += 1
            print(f"\n{'='*70}")
            print(f"[{eval_count}/{total_evals}] {model_id} @ {checkpoint}")
            print(f"{'='*70}")
            
            result = run_single_evaluation(
                model_id=model_id,
                checkpoint=checkpoint,
                tasks=task_names,
                output_base=output_base,
                max_new_tokens=args.max_new_tokens,
                load_vllm=args.load_vllm,
                num_shots=args.num_shots,
                skip_existing=not args.force_reeval,
                eval_mode=args.eval_mode,
                spaced=args.spaced,
                quantization=args.quantization,
            )
            all_results.append(result)
    
    # 建立summary DataFrames
    print("\n" + "="*70)
    print("CREATING SUMMARY")
    print("="*70)
    
    # Count cached vs 新的 evaluations
    total_evals = sum(len(r['results']) for r in all_results)
    cached_evals = sum(1 for r in all_results for t in r['results'] if t.get('cached', False))
    new_evals = total_evals - cached_evals
    
    print(f"\n📊 Evaluation Statistics:")
    print(f"  Total evaluations: {total_evals}")
    print(f"  ✓ Cached (skipped): {cached_evals}")
    print(f"  🔄 Newly evaluated: {new_evals}")
    print(f"  Time saved: ~{cached_evals * 3} minutes (est.)")
    
    # Flatten 結果 for detailed CSV
    detailed_rows = []
    for eval_result in all_results:
        for task_result in eval_result['results']:
            row = {
                'model': eval_result['model_id'],
                'checkpoint': eval_result['checkpoint'],
                'task': task_result['task'],
                'success': task_result['success'],
                'error': task_result['error'],
                'cached': task_result.get('cached', False),
            }
            # Add all metrics
            if task_result['success'] and task_result['metrics']:
                row.update(task_result['metrics'])
            detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # 建立task-特定 suffix for 輸出 檔案 to avoid overwriting
    # When running single 任務 (common in array jobs), include 任務 名稱 in filename
    if len(task_names) == 1:
        task_suffix = f"_{sanitize_task_name(task_names[0])}"
    else:
        task_suffix = ""
    
    detailed_path = output_base / f"detailed_results{task_suffix}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # 建立pivot table for easy comparison (準確率)
    if 'accuracy' in detailed_df.columns:
        pivot_df = detailed_df.pivot_table(
            index=['model', 'checkpoint'],
            columns='task',
            values='accuracy',
            aggfunc='first'
        )
        pivot_path = output_base / f"accuracy_pivot{task_suffix}.csv"
        pivot_df.to_csv(pivot_path)
        print(f"✓ Accuracy pivot table saved to: {pivot_path}")
        
        # Print 摘要
        print("\n" + "="*70)
        print("ACCURACY SUMMARY")
        print("="*70)
        print(pivot_df.to_string())
    
    # 建立pivot table for loss (continuous metrics)
    if 'mean_loss' in detailed_df.columns:
        loss_pivot_df = detailed_df.pivot_table(
            index=['model', 'checkpoint'],
            columns='task',
            values='mean_loss',
            aggfunc='first'
        )
        loss_pivot_path = output_base / f"loss_pivot{task_suffix}.csv"
        loss_pivot_df.to_csv(loss_pivot_path)
        print(f"✓ Loss pivot table saved to: {loss_pivot_path}")
        
        # Print 摘要
        print("\n" + "="*70)
        print("LOSS SUMMARY (lower is better)")
        print("="*70)
        print(loss_pivot_df.to_string())
    
    # Success 摘要
    success_df = detailed_df.groupby(['model', 'checkpoint'])['success'].agg(['sum', 'count'])
    success_df['success_rate'] = success_df['sum'] / success_df['count']
    
    print("\n" + "="*70)
    print("EVALUATION SUCCESS RATE")
    print("="*70)
    print(success_df.to_string())
    
    print(f"\n💾 All results saved to: {output_base}")
    
    # 生成plots if matplotlib is 可用
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive 後端
        from scripts.plot_checkpoint_results import prepare_plot_data, plot_all_tasks_overview, plot_task_progression
        
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        plot_dir = output_base / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Prepare 資料
        plot_df = detailed_df[detailed_df['success'] == True].copy()
        
        if len(plot_df) > 0:
            plot_df = prepare_plot_data(plot_df)
            tasks = plot_df['task'].unique()
            
            # === 準確率 PLOTS ===
            if 'accuracy' in plot_df.columns:
                print("\n📊 Creating accuracy plots...")
                
                # Overview plot
                overview_path = plot_dir / "overview_all_tasks_accuracy.png"
                plot_all_tasks_overview(
                    df=plot_df,
                    metric='accuracy',
                    output_path=overview_path,
                    figsize=(16, 12),
                    style='seaborn',
                )
                
                # Individual 任務 plots
                print(f"  Creating {len(tasks)} individual accuracy plots...")
                for task in tasks:
                    task_path = plot_dir / f"{task}_accuracy.png"
                    plot_task_progression(
                        df=plot_df,
                        task=task,
                        metric='accuracy',
                        output_path=task_path,
                        figsize=(12, 8),
                        style='seaborn',
                    )
                    matplotlib.pyplot.close('all')
            
            # === LOSS PLOTS (continuous metrics) ===
            if 'mean_loss' in plot_df.columns:
                print("\n📊 Creating loss plots...")
                
                # Overview plot for loss
                loss_overview_path = plot_dir / "overview_all_tasks_loss.png"
                plot_all_tasks_overview(
                    df=plot_df,
                    metric='mean_loss',
                    output_path=loss_overview_path,
                    figsize=(16, 12),
                    style='seaborn',
                )
                
                # Individual 任務 loss plots
                print(f"  Creating {len(tasks)} individual loss plots...")
                for task in tasks:
                    task_path = plot_dir / f"{task}_loss.png"
                    plot_task_progression(
                        df=plot_df,
                        task=task,
                        metric='mean_loss',
                        output_path=task_path,
                        figsize=(12, 8),
                        style='seaborn',
                    )
                    matplotlib.pyplot.close('all')
            
            # === PERPLEXITY PLOTS ===
            if 'mean_perplexity' in plot_df.columns:
                print("\n📊 Creating perplexity plots...")
                
                # Overview plot for perplexity
                ppl_overview_path = plot_dir / "overview_all_tasks_perplexity.png"
                plot_all_tasks_overview(
                    df=plot_df,
                    metric='mean_perplexity',
                    output_path=ppl_overview_path,
                    figsize=(16, 12),
                    style='seaborn',
                )
                
                # Individual 任務 perplexity plots
                print(f"  Creating {len(tasks)} individual perplexity plots...")
                for task in tasks:
                    task_path = plot_dir / f"{task}_perplexity.png"
                    plot_task_progression(
                        df=plot_df,
                        task=task,
                        metric='mean_perplexity',
                        output_path=task_path,
                        figsize=(12, 8),
                        style='seaborn',
                    )
                    matplotlib.pyplot.close('all')
            
            print(f"\n✅ Plots saved to: {plot_dir}")
        else:
            print("\n⚠️  No successful evaluations found, skipping plots")
    
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plot generation")
        print("   Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"\n⚠️  Error generating plots: {e}")
        print("   Results are still saved in CSV format")


if __name__ == "__main__":
    main()
