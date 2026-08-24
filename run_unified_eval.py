#!/usr/bin/env python3
"""Command-line interface for the unified 任務 評估 system."""

import argparse
import json
from pathlib import Path
import sys
import os

# Add project root to 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.base_task import TaskConfig
from tasks.evaluator import TaskEvaluator, ModelConfig, EvaluationConfig
from tasks.registry import TaskRegistry


def create_model_config_from_args(args) -> ModelConfig:
    """建立ModelConfig from command 行 arguments."""
    return ModelConfig(
        model_id=args.model_id,
        backend=args.backend,
        checkpoint=args.checkpoint,
        local_path=args.local_path,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code
    )


def create_eval_config_from_args(args) -> EvaluationConfig:
    """建立EvaluationConfig from command 行 arguments."""
    return EvaluationConfig(
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_detailed_results=args.save_detailed_results,
        batch_size=args.batch_size,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay
    )


def main():
    # 檢查for 列表 類別 flag 第一個 (before full argument parsing)
    if "--list_textfrct_categories" in sys.argv:
        print_textfrct_categories()
        return 0
    
    parser = argparse.ArgumentParser(description="Unified Task Evaluation System")
    
    # 任務參數
    task_group = parser.add_argument_group("Task Configuration")
    
    # Discover 可用 任務 dynamically
    registry = TaskRegistry()
    available_tasks = list(registry.discover_tasks().keys())
    
    task_group.add_argument("--task_type", type=str, required=True,
                           choices=available_tasks,
                           help=f"Type of task to run. Available: {', '.join(available_tasks)}")
    
    # TextFRCT 特定 arguments (only if textfrct is 可用)
    if "textfrct" in available_tasks:
        task_group.add_argument("--skip_subjective", action="store_true",
                               help="Skip subjective categories (for TextFRCT)")
        
        textfrct_group = parser.add_argument_group("TextFRCT Configuration")
        textfrct_group.add_argument("--textfrct_categories", type=str, nargs="*",
                                   help="Specific TextFRCT categories to evaluate (e.g., CV1 CV2 FA1). "
                                        "If not specified, all categories are used. "
                                        "Common categories: CV1 (scrambled words), CV2 (hidden words), "
                                        "CV3 (incomplete words), FA1 (associations), FA2 (opposites), "
                                        "V1-V5 (vocabulary), RG1-RG3 (reasoning)")
        textfrct_group.add_argument("--list_textfrct_categories", action="store_true",
                                   help="List all available TextFRCT categories and exit")
    
    # 模型參數
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_id", type=str, required=True,
                            help="Model identifier")
    model_group.add_argument("--backend", type=str, required=True,
                            choices=["vllm", "transformers", "openai", "together"],
                            help="Model backend to use")
    model_group.add_argument("--checkpoint", type=str,
                            help="Model checkpoint/revision")
    model_group.add_argument("--local_path", type=str,
                            help="Local path to model (overrides model_id)")
    model_group.add_argument("--api_key", type=str,
                            help="API key for OpenAI/Together")
    
    # 生成參數
    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument("--temperature", type=float, default=0.0,
                          help="Generation temperature")
    gen_group.add_argument("--max_tokens", type=int, default=100,
                          help="Maximum tokens to generate")
    gen_group.add_argument("--top_p", type=float, default=1.0,
                          help="Top-p sampling parameter")
    gen_group.add_argument("--tensor_parallel_size", type=int,
                          help="Tensor parallel size for vLLM")
    gen_group.add_argument("--trust_remote_code", action="store_true", default=True,
                          help="Trust remote code for model loading")
    
    # 評估參數
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument("--output_dir", type=str, default="results",
                           help="Output directory for results")
    eval_group.add_argument("--save_predictions", action="store_true", default=True,
                           help="Save prediction results")
    eval_group.add_argument("--save_detailed_results", action="store_true", default=True,
                           help="Save detailed results with prompts and predictions")
    eval_group.add_argument("--batch_size", type=int, default=1,
                           help="Batch size for evaluation")
    eval_group.add_argument("--retry_attempts", type=int, default=3,
                           help="Number of retry attempts for API calls")
    eval_group.add_argument("--retry_delay", type=float, default=1.0,
                           help="Delay between retry attempts")
    
    args = parser.parse_args()
    
    # 建立設定
    model_config = create_model_config_from_args(args)
    eval_config = create_eval_config_from_args(args)
    
    # 建立task dynamically using registry - NO HARDCODED CONDITIONALS!
    print(f"Creating {args.task_type} task...")
    
    # 取得task 類別 from registry
    task_class = registry.get_task_class(args.task_type)
    if not task_class:
        print(f"Error: Task '{args.task_type}' not found in registry")
        return 1
    
    # 建立task-特定 設定
    if args.task_type == "textfrct":
        # TextFRCT needs special 設定 and constructor arguments
        config = TaskConfig(
            name="textfrct_cli",
            description="TextFRCT evaluation from CLI",
            data_path="dataset/TextFRCT.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
            metadata={
                "skip_subjective": getattr(args, 'skip_subjective', False),
                "categories": getattr(args, 'textfrct_categories', None)
            }
        )
        task = task_class(
            config,
            skip_subjective=getattr(args, 'skip_subjective', False),
            categories=getattr(args, 'textfrct_categories', None)
        )
        if hasattr(args, 'textfrct_categories') and args.textfrct_categories:
            print(f"Filtering to categories: {args.textfrct_categories}")
        
    elif args.task_type == "basic_arithmetic":
        # BasicArithmetic uses 記憶體內 資料
        config = TaskConfig(
            name="basic_arithmetic_cli",
            description="Basic arithmetic evaluation from CLI",
            data_format="memory",
            data_path=None,
            in_memory_data=None,  # Signal to use 任務's 預設data
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"]
        )
        task = task_class(config)
    elif args.task_type == "ioi_task":
        # IOITask uses mib-bench/ioi 資料集
        config = TaskConfig(
            name="ioi_task_cli",
            description="Indirect Object Identification (IOI) evaluation from mib-bench/ioi dataset",
            data_format="huggingface",
            data_path="mib-bench/ioi",
            input_column="prompt",  # The incomplete sentence
            output_column="choices",  # 列表 of choices, use with answer_index
            evaluation_metrics=["accuracy"]
        )
        task = task_class(config)
    else:
        # Generic 任務 creation for other 任務
        config = TaskConfig(
            name=f"{args.task_type}_cli",
            description=f"{args.task_type} evaluation from CLI",
            data_path=f"dataset/{args.task_type}.csv",  # 預設 資料 路徑
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"]
        )
        task = task_class(config)
    
    print(f"✅ Successfully created {args.task_type} task")
    
    # 建立evaluator and run 評估
    print("Creating evaluator...")
    evaluator = TaskEvaluator(model_config, eval_config)
    print("Running evaluation...")
    results = evaluator.evaluate_task(task)
    
    # 列印摘要結果
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Task: {results['task_name']}")
    print(f"Model: {results['model_id']}")
    print(f"Backend: {results['backend']}")
    print(f"Examples: {results['num_examples']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\nDetailed results saved to: {eval_config.output_dir}")
    
    return 0


def print_textfrct_categories():
    """Print all 可用 TextFRCT 類別 with descriptions."""
    print("Available TextFRCT Categories:")
    print("=" * 50)
    
    categories = {
        "CV1": "Scrambled Words - Unscramble letters to form words",
        "CV2": "Hidden Words - Find words hidden in letter strings", 
        "CV3": "Incomplete Words - Complete words with missing letters",
        "FA1": "Controlled Association - Generate related words",
        "FA2": "Opposites - Generate antonyms",
        "FA3": "Figures of Speech - Complete metaphors/similes",
        "FE1": "Making Sentences - Create sentences from patterns",
        "FE2": "Arranging Words - Form sentences from word lists",
        "FE3": "Rewriting - Rewrite sentences with same meaning",
        "FI1": "Topics Test - Generate ideas about topics",
        "FI2": "Theme Test - Write paragraphs about themes", 
        "FI3": "Thing Categories - List items in categories",
        "FW1": "Word Fluency - Generate words with constraints",
        "FW2": "First and Last Letters - Words starting/ending with letters",
        "FW3": "Pattern Words - Words matching letter patterns",
        "I1": "Letter Sets - Pattern recognition with letters",
        "I2": "Figure Classification - Spatial reasoning",
        "MA2": "Object-Number - Associate objects with numbers",
        "MA3": "First Names - Recall associated names",
        "RG1": "Arithmetic Aptitude - Basic math operations",
        "RG2": "Number Series - Continue number sequences",
        "RG3": "Mathematical Reasoning - Solve math word problems",
        "RL1": "Syllogistic Reasoning - Logical syllogisms",
        "RL3": "Necessary Inference - Required logical conclusions",
        "RL4": "Language Deciphering - Decode artificial languages",
        "V1": "Vocabulary Level 1 - Basic vocabulary",
        "V2": "Vocabulary Level 2 - Intermediate vocabulary",
        "V3": "Vocabulary Level 3 - Advanced vocabulary", 
        "V4": "Vocabulary Level 4 - Expert vocabulary",
        "V5": "Vocabulary Level 5 - Scholar vocabulary",
        "XU1": "Object Combination - Combine objects creatively",
        "XU2": "Substitute Uses - Alternative uses for objects",
        "XU3": "Object Grouping - Group objects by function",
        "XU4": "Different Uses - Creative uses for objects"
    }
    
    # 群組 by 主要 類別
    groups = {
        "CV (Convergent Visual)": ["CV1", "CV2", "CV3"],
        "FA (Fluent Associational)": ["FA1", "FA2", "FA3"],
        "FE (Flexible Expression)": ["FE1", "FE2", "FE3"],
        "FI (Fluent Ideational)": ["FI1", "FI2", "FI3"],
        "FW (Fluent Word)": ["FW1", "FW2", "FW3"],
        "I (Inductive)": ["I1", "I2"],
        "MA (Memory/Association)": ["MA2", "MA3"],
        "RG (Reasoning)": ["RG1", "RG2", "RG3"],
        "RL (Reasoning/Logic)": ["RL1", "RL3", "RL4"],
        "V (Vocabulary)": ["V1", "V2", "V3", "V4", "V5"],
        "XU (Creative)": ["XU1", "XU2", "XU3", "XU4"]
    }
    
    for group_name, group_categories in groups.items():
        print(f"\n{group_name}:")
        for cat in group_categories:
            if cat in categories:
                print(f"  {cat}: {categories[cat]}")
    
    print(f"\nExample usage:")
    print(f"  # Test only scrambled words and vocabulary:")
    print(f"  python run_unified_eval.py --task_type textfrct --textfrct_categories CV1 V1 V2")
    print(f"  ")
    print(f"  # Test only reasoning tasks:")
    print(f"  python run_unified_eval.py --task_type textfrct --textfrct_categories RG1 RG2 RG3 RL1")
    print(f"  ")
    print(f"  # Test all categories (default):")
    print(f"  python run_unified_eval.py --task_type textfrct")


if __name__ == "__main__":
    exit(main())
