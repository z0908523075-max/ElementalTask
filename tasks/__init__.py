"""任務 評估 system with 自動 任務 發現."""

from .base_task import BaseTask, TaskConfig
from .evaluator import TaskEvaluator, ModelConfig, EvaluationConfig
from .registry import (
    discover_tasks, 
    register_task, 
    get_task_class, 
    get_task, 
    list_tasks, 
    get_task_info
)

# 自動地 discover and 註冊 all 任務
print("Discovering tasks...")
discover_tasks()

__all__ = [
    "BaseTask", 
    "TaskConfig", 
    "TaskEvaluator", 
    "ModelConfig", 
    "EvaluationConfig",
    "discover_tasks",
    "register_task",
    "get_task_class",
    "get_task",
    "list_tasks",
    "get_task_info"
]
