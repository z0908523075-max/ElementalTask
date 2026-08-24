"""自動 任務 發現 and 註冊 system."""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List
from .base_task import BaseTask, TaskConfig


class TaskRegistry:
    """自動 任務 發現 and 註冊 system."""

    # Legacy 組合式 任務 名稱 now backed by blended 組合 資料.
    _COMPOSITIONAL_TO_BLENDED_ALIASES = {
        "coref_tracking_query",
        "decipher_apply_reason",
        "extract_verify",
        "opplan_solve",
    }
    
    def __init__(self):
        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._discovered = False
    
    def discover_tasks(self, implementations_path: str = None) -> Dict[str, Type[BaseTask]]:
        """自動地 discover and 註冊 任務 from the 實作 目錄."""
        if self._discovered:
            return self._tasks
        
        if implementations_path is None:
            # 預設 to 任務/實作 目錄
            current_dir = Path(__file__).parent
            implementations_path = current_dir / "implementations"
        else:
            implementations_path = Path(implementations_path)
        
        if not implementations_path.exists():
            print(f"Warning: Task implementations directory not found: {implementations_path}")
            self._discovered = True
            return self._tasks
        
        # 取得all Python 檔案 in the 實作 目錄
        python_files = list(implementations_path.glob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]
        
        discovered_count = 0
        failed_count = 0
        
        for py_file in python_files:
            try:
                # 轉換file 路徑 to module 名稱
                module_name = f"tasks.implementations.{py_file.stem}"
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find all 類別 that inherit from BaseTask
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseTask) and 
                        obj != BaseTask and 
                        hasattr(obj, 'TASK_NAME')):
                        
                        task_name = obj.TASK_NAME
                        if task_name in self._tasks:
                            print(f"Warning: Task '{task_name}' already registered, skipping duplicate from {py_file.name}")
                        else:
                            self._tasks[task_name] = obj
                            discovered_count += 1
                            print(f"✅ Registered task: '{task_name}' from {py_file.name}")
            
            except ImportError as e:
                print(f"⚠️  Could not import {py_file.name}: {e}")
                failed_count += 1
            except Exception as e:
                print(f"❌ Error processing {py_file.name}: {e}")
                failed_count += 1
        
        self._discovered = True
        print(f"\n📊 Task Discovery Summary:")
        print(f"  ✅ Successfully registered: {discovered_count} tasks")
        if failed_count > 0:
            print(f"  ⚠️  Failed to load: {failed_count} files")
        print(f"  📋 Available tasks: {list(self._tasks.keys())}")
        return self._tasks
    
    def register_task(self, name: str, task_class: Type[BaseTask]) -> None:
        """Manually 註冊 a 任務 類別."""
        if not issubclass(task_class, BaseTask):
            raise ValueError(f"Task class must inherit from BaseTask")
        
        if name in self._tasks:
            print(f"Warning: Overriding existing task '{name}'")
        
        self._tasks[name] = task_class
        print(f"✅ Manually registered task: '{name}'")
    
    def get_task_class(self, name: str) -> Type[BaseTask]:
        """取得a 任務 類別 by 名稱."""
        if not self._discovered:
            self.discover_tasks()
        
        if name not in self._tasks:
            available = list(self._tasks.keys())
            raise ValueError(f"Task '{name}' not found. Available tasks: {available}")
        
        return self._tasks[name]
    
    def create_task(self, name: str, spaced: bool = False, *args, **kwargs) -> BaseTask:
        """建立一個task 實例 by 名稱.
        
        If no 設定/args are provided, tries to use 工廠函式s for proper 預設.
        Falls back to 建立中 a basic TaskConfig if no factory is 可用.
        
        參數：
            名稱: 任務 名稱, optionally with subtask 篩選 (e.g., 'compositional:upper_reverse')
            spaced: If True, use spaced version of 任務 (adds spaces between 字元)
            *args, **kwargs: Additional arguments passed to 任務 constructor
        """
        from .base_task import TaskConfig
        
        # Allow a 簡單 subtask syntax: '任務:sub1,sub2' -> 基礎 任務 with 類別 filters
        base_name = name
        subtask_specs = None
        if isinstance(name, str) and ':' in name:
            base_name, spec = name.split(':', 1)
            subtask_specs = spec  # Keep as 字串 for 工廠函式s

        # If arguments are provided, use them directly (honor explicit kwargs)
        if args or kwargs:
            task_class = self.get_task_class(base_name)
            if not args and 'config' not in kwargs:
                kwargs['config'] = TaskConfig(name=base_name)
            task_instance = task_class(*args, **kwargs)
            # If subtasks were requested, try to apply a 簡單 post-filter
            if subtask_specs:
                try:
                    # 轉換string to 列表 for filtering
                    filter_list = [s.strip() for s in subtask_specs.split(',') if s.strip()]
                    import pandas as pd
                    if hasattr(task_instance, 'data') and isinstance(task_instance.data, pd.DataFrame):
                        if 'category_id' in task_instance.data.columns:
                            task_instance.data = task_instance.data[task_instance.data['category_id'].isin(filter_list)]
                        elif 'category_name' in task_instance.data.columns:
                            task_instance.data = task_instance.data[task_instance.data['category_name'].isin(filter_list)]
                    elif hasattr(task_instance, 'data') and isinstance(task_instance.data, list):
                        task_instance.data = [r for r in task_instance.data if r.get('category_id') in filter_list or r.get('category_name') in filter_list]
                except Exception:
                    pass
            return task_instance
        
        # Try to use 工廠函式s for 任務 that need special initialization
        factory_map = {
            'copying': lambda: self._import_and_call('tasks.implementations.copying_task', 'make_copying_task', use_generator=True),
            'ignoring_context': lambda: self._import_and_call('tasks.implementations.ignoring_context_task', 'make_ignoring_context_task', use_generator=True),
            'string_analogy': lambda: self._import_and_call('tasks.implementations.string_analogy_task', 'make_string_analogy_task', use_generator=True),
            'basic_arithmetic': lambda: self._import_and_call('tasks.implementations.basic_arithmetic', 'create_basic_arithmetic_task'),
            'ioi_task': lambda: self._import_and_call('tasks.implementations.ioi_task', 'create_ioi_task'),
            'token_reversal': lambda: self._import_and_call('tasks.implementations.token_reversal', 'create_token_reversal_task'),
            'part_of_speech': lambda: self._import_and_call('tasks.implementations.pos_id', 'create_pos_task'),
            'textfrct': lambda: self._create_textfrct_task(subtask_specs),
            'simple_icl': lambda: self._create_simple_icl_task(),
            'simple': lambda: self._create_simple_task(),
            'math': lambda: self._create_math_task(),
            'compositional': lambda: self._create_compositional_task(subtask_specs, spaced=spaced),
            'multistep_arithmetic': lambda: self._import_and_call('tasks.implementations.multistep_arithmetic_task', 'create_multistep_arithmetic_task', category=subtask_specs),
            'logical_ops': lambda: self._import_and_call('tasks.implementations.logical_ops_task', 'create_logical_ops_task', category=subtask_specs),
            'fact_extraction': lambda: self._import_and_call('tasks.implementations.fact_extraction_task', 'create_fact_extraction_task', category=subtask_specs),
            'coreference': lambda: self._import_and_call('tasks.implementations.coreference_task', 'create_coreference_task', category=subtask_specs),
            'word_problem_single_step': lambda: self._import_and_call('tasks.implementations.math_word_problem_tasks', 'create_word_problem_single_step_task', category=subtask_specs),
            'quantity_comparison': lambda: self._import_and_call('tasks.implementations.math_word_problem_tasks', 'create_quantity_comparison_task', category=subtask_specs),
            'rate_unit': lambda: self._import_and_call('tasks.implementations.math_word_problem_tasks', 'create_rate_unit_task', category=subtask_specs),
            'multi_entity_tracking': lambda: self._import_and_call('tasks.implementations.math_word_problem_tasks', 'create_multi_entity_tracking_task', category=subtask_specs),
            'benchmark': lambda: self._create_benchmark_task(subtask_specs),
        }
        
        if base_name in factory_map:
            try:
                task_instance = factory_map[base_name]()
                # Note: 組合式 and textfrct handle their own filtering via factory args
                # For other 任務, apply post-filtering if subtasks were requested
                if subtask_specs and base_name not in ('compositional', 'textfrct'):
                    try:
                        filter_list = [s.strip() for s in subtask_specs.split(',') if s.strip()]
                        import pandas as pd
                        if hasattr(task_instance, 'data') and isinstance(task_instance.data, pd.DataFrame):
                            if 'category_id' in task_instance.data.columns:
                                task_instance.data = task_instance.data[task_instance.data['category_id'].isin(filter_list)]
                            elif 'category_name' in task_instance.data.columns:
                                task_instance.data = task_instance.data[task_instance.data['category_name'].isin(filter_list)]
                        elif hasattr(task_instance, 'data') and isinstance(task_instance.data, list):
                            task_instance.data = [r for r in task_instance.data if r.get('category_id') in filter_list or r.get('category_name') in filter_list]
                    except Exception:
                        pass
                return task_instance
            except Exception as e:
                print(f"Warning: Factory function failed for '{name}': {e}")
                print(f"Falling back to default config...")
        
        # Fallback: 建立with minimal 設定
        task_class = self.get_task_class(base_name)
        return task_class(config=TaskConfig(name=name))
    
    def _import_and_call(self, module_name: str, function_name: str, **kwargs):
        """Import a module and call a function from it."""
        import importlib
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func(**kwargs)
    
    def _create_simple_icl_task(self):
        """建立一個SimpleICLTask with 資料 from simple.csv."""
        config = TaskConfig(
            name="simple_icl",
            description="Simple ICL task with category-based demonstrations",
            data_path="dataset/simple.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("simple_icl")
        return task_class(config)
    
    def _create_simple_task(self):
        """建立一個SimpleTask with 資料 from simple.csv."""
        config = TaskConfig(
            name="simple",
            description="Simple task with exact match evaluation",
            data_path="dataset/simple.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("simple")
        return task_class(config)
    
    def _create_math_task(self):
        """建立一個MathTask (生成 合成 資料)."""
        config = TaskConfig(
            name="math",
            description="Math task with synthetic arithmetic problems",
            data_format="generator",  # Indicates 合成/已生成 資料
            input_column="input",
            output_column="output",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("math")
        return task_class(config)
    
    def _create_compositional_task(self, category_filter=None, spaced=False):
        """建立一個CompositionalTask with 可選category filtering and spacing.
        
        支援:
            - 組合式 (all 類別)
            - 組合式:upper_reverse (single 類別)
            - 組合式:upper_reverse,lower_first (multiple 類別)
            - spaced=True for character-spaced 輸入/輸出
        """
        if category_filter:
            categories = [c.strip() for c in category_filter.split(",") if c.strip()]
            if categories and all(c in self._COMPOSITIONAL_TO_BLENDED_ALIASES for c in categories):
                from tasks.implementations.blended_compositions_task import create_blended_compositions_task

                # Keep incoming namespace stable while routing to blended 資料.
                blended_name = f"blended_compositions:{','.join(categories)}"
                return create_blended_compositions_task(name=blended_name)

        task_name = "compositional"
        if category_filter:
            task_name = f"compositional:{category_filter}"
        if spaced:
            task_name += "_spaced"
        
        config = TaskConfig(
            name=task_name,
            description="Compositional task with chained string operations",
            data_path="dataset/compositional_spaced.csv" if spaced else "dataset/compositional.csv",
            data_format="csv",
            input_column="input",
            output_column="output",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("compositional")
        task = task_class(config, spaced=spaced)
        
        # 篩選 by 類別 if specified
        if category_filter:
            categories = [c.strip() for c in category_filter.split(",")]
            if hasattr(task, 'data') and task.data:
                if isinstance(task.data, list):
                    task.data = [d for d in task.data if d.get('category_name') in categories]

            # Fallback: 支援 blended 組合 via 組合式:<類別>
            # so they remain discoverable under the 組合式 namespace.
            if (not getattr(task, 'data', None)) and len(categories) == 1:
                blended_category = categories[0]
                try:
                    return self._import_and_call(
                        'tasks.implementations.blended_compositions_task',
                        'create_blended_compositions_task',
                        category=blended_category,
                        name=f"compositional:{blended_category}",
                    )
                except Exception:
                    pass
        
        return task
    
    def _create_textfrct_task(self, category_filter=None):
        """建立一個TextFRCT 任務 with 可選category filtering.
        
        支援:
            - textfrct (all 類別)
            - textfrct:CV1 (single 類別)
            - textfrct:CV1,CV2,FA1 (multiple 類別)
        """
        from tasks.implementations.textfrct_task import TextFRCTTask
        
        # Parse 類別 篩選 (can be single 類別 or comma-separated 列表)
        categories = None
        if category_filter:
            if isinstance(category_filter, list):
                categories = category_filter
            else:
                categories = [c.strip() for c in str(category_filter).split(",")]
        
        task_name = f"textfrct:{','.join(categories)}" if categories else "textfrct"
        
        config = TaskConfig(
            name=task_name,
            description="TextFRCT evaluation dataset",
            data_path="dataset/TextFRCT.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        
        task = TextFRCTTask(config, skip_subjective=True, categories=categories)
        
        # 篩選 資料 by category_id if 類別 specified
        if categories and hasattr(task, 'data') and task.data:
            if isinstance(task.data, list):
                task.data = [d for d in task.data if d.get('category_id') in categories]
            
        return task
    
    def _create_benchmark_task(self, task_name: str):
        """建立一個benchmark 任務 (ARC-Challenge, BoolQ, WinoGrande, GSM8K) by 名稱.

        支援:
            benchmark:arc_challenge
            benchmark:boolq
            benchmark:winogrande
            benchmark:gsm8k
        """
        from tasks.benchmarks.lmeval_tasks import make_benchmark_task
        return make_benchmark_task(task_name)

    def list_tasks(self) -> List[str]:
        """列表 all 可用 任務 名稱."""
        if not self._discovered:
            self.discover_tasks()
        return list(self._tasks.keys())
    
    def get_task_info(self, name: str = None) -> Dict:
        """取得information about 任務."""
        if not self._discovered:
            self.discover_tasks()
        
        if name is None:
            # 回傳info for all 任務
            return {
                task_name: {
                    "class": task_class.__name__,
                    "module": task_class.__module__,
                    "docstring": task_class.__doc__ or "No description available"
                }
                for task_name, task_class in self._tasks.items()
            }
        else:
            # 回傳info for 特定 任務
            if name not in self._tasks:
                raise ValueError(f"Task '{name}' not found")
            
            task_class = self._tasks[name]
            return {
                "name": name,
                "class": task_class.__name__,
                "module": task_class.__module__,
                "docstring": task_class.__doc__ or "No description available",
                "methods": [method for method in dir(task_class) if not method.startswith('_')]
            }


# Global registry 實例
_task_registry = TaskRegistry()

# Public API functions
def discover_tasks(implementations_path: str = None) -> Dict[str, Type[BaseTask]]:
    """Discover and 註冊 all 任務 from the 實作 目錄."""
    return _task_registry.discover_tasks(implementations_path)

def register_task(name: str, task_class: Type[BaseTask]) -> None:
    """Manually 註冊 a 任務 類別."""
    _task_registry.register_task(name, task_class)

def get_task_class(name: str) -> Type[BaseTask]:
    """取得a 任務 類別 by 名稱."""
    return _task_registry.get_task_class(name)

def get_task(name: str, spaced: bool = False, *args, **kwargs) -> BaseTask:
    """建立一個task 實例 by 名稱.
    
    參數：
        名稱: 任務 名稱, optionally with subtask 篩選 (e.g., 'compositional:upper_reverse')
        spaced: If True, use spaced version (spaces between 字元 in 輸入/輸出)
        *args, **kwargs: Additional arguments passed to 任務 constructor
    
    回傳：
        任務 實例
    
    範例：
        >>> task = get_task("compositional:upper_reverse")
        >>> task = get_task("compositional:upper_reverse", spaced=True)
        >>> task = get_task("textfrct:CV1")
    """
    return _task_registry.create_task(name, spaced=spaced, *args, **kwargs)

def list_tasks() -> List[str]:
    """列表 all 可用 任務 名稱."""
    return _task_registry.list_tasks()

def get_task_info(name: str = None) -> Dict:
    """取得information about 任務."""
    return _task_registry.get_task_info(name)

def list_all_tasks(include_compositional: bool = True, include_textfrct: bool = True, include_simple_icl: bool = True) -> Dict[str, List[str]]:
    """列表 all 可用 任務 including subtasks.
    
    參數：
        include_compositional: If True, enumerate all 組合式 subtasks
        include_textfrct: If True, enumerate all TextFRCT subtasks
        include_simple_icl: If True, enumerate all simple_icl 類別
    
    回傳：
        字典 mapping 任務 類別 to 列表 of 任務 名稱
    """
    if not _task_registry._discovered:
        _task_registry.discover_tasks()
    
    result = {
        "base_tasks": [],
        "compositional_tasks": [],
        "textfrct_tasks": [],
        "simple_icl_tasks": [],
    }
    
    for task_name in sorted(_task_registry._tasks.keys()):
        if task_name == "compositional" and include_compositional:
            # Add 基礎 組合式 任務
            result["base_tasks"].append("compositional")
            
            # Enumerate all 組合式 subtasks
            try:
                from tasks.implementations.compositional_task import STRING_COMPOSITIONS, LOOKUP_COMPOSITIONS
                
                # Add all 字串組合
                for comp_name in sorted(STRING_COMPOSITIONS.keys()):
                    result["compositional_tasks"].append(f"compositional:{comp_name}")
                    result["compositional_tasks"].append(f"compositional:{comp_name} (spaced)")
                
                # Add all 基於查找的 組合
                for comp_name in sorted(LOOKUP_COMPOSITIONS.keys()):
                    result["compositional_tasks"].append(f"compositional:{comp_name}")
                    result["compositional_tasks"].append(f"compositional:{comp_name} (spaced)")

                # Add blended 組合 類別 under 組合式 namespace
                try:
                    from tasks.implementations.blended_compositions_task import BlendedCompositionsTask
                    for comp_name in sorted(BlendedCompositionsTask.CATEGORY_DATA.keys()):
                        result["compositional_tasks"].append(f"compositional:{comp_name}")
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"Warning: Could not enumerate compositional tasks: {e}")
        
        elif task_name == "textfrct" and include_textfrct:
            # Add 基礎 textfrct 任務
            result["base_tasks"].append("textfrct")
            
            # Enumerate all TextFRCT 類別
            try:
                import pandas as pd
                from pathlib import Path
                csv_path = Path(__file__).parent.parent / "dataset" / "TextFRCT.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    categories = sorted(df["category_id"].unique())
                    for cat in categories:
                        result["textfrct_tasks"].append(f"textfrct:{cat}")
            except Exception as e:
                print(f"Warning: Could not enumerate TextFRCT tasks: {e}")
        
        elif task_name == "simple_icl" and include_simple_icl:
            # Add 基礎 simple_icl 任務
            result["base_tasks"].append("simple_icl")
            
            # Enumerate all simple_icl 類別
            try:
                import pandas as pd
                from pathlib import Path
                csv_path = Path(__file__).parent.parent / "dataset" / "simple.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    categories = sorted(df["category_name"].unique())
                    for cat in categories:
                        result["simple_icl_tasks"].append(f"simple_icl:{cat}")
            except Exception as e:
                print(f"Warning: Could not enumerate simple_icl tasks: {e}")
        
        else:
            # Regular 任務
            result["base_tasks"].append(task_name)
    
    return result

def print_all_tasks():
    """Print a 格式化 列表 of all 可用 任務."""
    tasks = list_all_tasks()
    
    print("\n" + "="*70)
    print("AVAILABLE TASKS")
    print("="*70)
    
    print("\n📋 BASE TASKS:")
    for task in tasks["base_tasks"]:
        print(f"  • {task}")
    
    if tasks["simple_icl_tasks"]:
        print(f"\n🎯 SIMPLE_ICL TASKS ({len(tasks['simple_icl_tasks'])} total):")
        for task in tasks["simple_icl_tasks"]:
            print(f"  • {task}")
    
    if tasks["compositional_tasks"]:
        print(f"\n🔗 COMPOSITIONAL TASKS ({len(tasks['compositional_tasks'])} total):")
        for task in tasks["compositional_tasks"]:
            print(f"  • {task}")
    
    if tasks["textfrct_tasks"]:
        print(f"\n📝 TEXTFRCT TASKS ({len(tasks['textfrct_tasks'])} total):")
        for task in tasks["textfrct_tasks"]:
            print(f"  • {task}")
    
    print("\n" + "="*70)
    total = (len(tasks["base_tasks"]) + len(tasks["compositional_tasks"]) + 
             len(tasks["textfrct_tasks"]) + len(tasks["simple_icl_tasks"]))
    print(f"TOTAL: {total} tasks available")
    print("="*70 + "\n")
