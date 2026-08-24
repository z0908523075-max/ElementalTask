"""
This 目錄 contains all 任務 實作 for 自動 發現.

Each 任務 實作 should:
1. Inherit from BaseTask
2. Have a unique 類別 名稱 ending in 'Task'
3. Define TASK_NAME as a 類別 attribute
4. Be importable from its module

範例：
```python
from tasks.base_task import BaseTask, TaskConfig

class MyCustomTask(BaseTask):
    TASK_NAME = "my_custom"  # Will be registered as "my_custom"
    
    def evaluate(self, predictions, split="test", **kwargs):
        # Implementation here
        pass
```

The 任務 registry will 自動地 discover and 註冊 all 任務 in this 目錄.
"""

from .antonym_task import AntonymTask
from .copying_task import CopyingTask
from .ignoring_context_task import IgnoringContextTask
from .string_analogy_task import StringAnalogyTask
