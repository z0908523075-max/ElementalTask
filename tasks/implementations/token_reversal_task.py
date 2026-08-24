from tasks.base_task import BaseTask, TaskConfig

class TokenReversalTask(BaseTask):
    """
    A 簡單 任務 that reverses the 輸入 字串.
    """
    TASK_NAME = "token_reversal"
    
    def __init__(self, config: TaskConfig, examples=None):
        super().__init__(config, examples=examples)

    def _evaluate(self, input_str: str) -> str:
        return input_str[::-1]

def make_token_reversal_task(config: TaskConfig, examples=None):
    if examples is None:
        examples = [
            {"input": "hello", "output": "olleh"},
            {"input": "world", "output": "dlrow"},
            {"input": "abc123", "output": "321cba"},
        ]
    return TokenReversalTask(config, examples=examples)
