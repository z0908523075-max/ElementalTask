"""Copying for testing induction heads.

This task presents examples where input equals output, testing the model's
ability to perform exact copying - a key capability of induction heads.

Two modes of operation:

1. **static mode** (use_generator=False):
   - Uses a fixed set of examples (default: 10 simple word/string)
   - ICL examples are sampled from this static set
   - Good for reproducibility and consistency
   
2. **Generator mode** (use_generator=True):
   - Generate unlimited random character sequences on the fly
   - Each call to get_icl_examples() produces a new random string
   - Configurable length range and character set
   - Perfect for testing with diverse inputs without data limitations

Example usage:
    # static mode
    task = make_copying_task(use_generator=False)
    example = task.get_icl_examples(num_examples=5)
    
    # Generator mode with custom parameters
    task = make_copying_task(use_generator=True, min_length=3, max_length=8)
    example = task.get_icl_examples(num_examples=10, charset="abc123", seed=42)
"""

import random
import string
from tasks.base_task import BaseTask, TaskConfig
from typing import Dict, List, Iterator


class CopyingTask(BaseTask):
    """
    A task where the output is an exact copy of the input.
    This tests induction head capabilities in language model.
    """
    TASK_NAME = "copying"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False, 
                 min_length: int = 3, max_length: int = 8):
        """
        Initialize the CopyingTask.
        
        Args: 
            configuration: a TaskConfig with task settings
            use_generator: If True, Generate random example on the fly instead of using static data
            min_length: Minimum length of generated random string (default: 3)
            max_length: Maximum length of generated random string (default: 8)
        """
        self.use_generator = use_generator
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(config)
    
    def generate_random_example(self, length: int = None, 
                               charset: str = None, 
                               seed: int = None) -> Dict[str, str]:
        """
        Generate a random copying example.
        
        Args: 
            length: Length of the random string. If None, randomly chosen between min_length and max_length
            charset: character set to use. If None, uses letters and digits
            seed: random seed for reproducibility
            
        Returns: 
            dictionary with 'input' and 'output' keys (both identical)
        """
        if seed is not None:
            random.seed(seed)
        
        if charset is None:
            charset = string.ascii_letters + string.digits
        
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        random_str = ''.join(random.choices(charset, k=length))
        return {"input": random_str, "output": random_str}
    
    def generate_examples(self, num_examples: int, 
                         charset: str = None,
                         seed: int = None) -> List[Dict[str, str]]:
        """
        Generate multiple random copying example.
        
        Args: 
            num_examples: number of examples to Generate
            charset: character set to use for Generate
            seed: random seed for reproducibility
            
        Returns: 
            list of example dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for _ in range(num_examples):
            examples.append(self.generate_random_example(charset=charset))
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: int = None,
        fresh: bool = True,
        charset: str = None,
    ) -> List[Dict[str, str]]:
        """
        Return examples in ICL format.
        
        If use_generator=True, Generate new random example each time.
        Otherwise, uses the standard BaseTask logic with stored data.
        
        Args: 
            num_examples: number of examples to return
            shuffle: Whether to shuffle (ignored if use_generator=True)
            seed: random seed for reproducibility
            fresh: Whether to prefer unused example (ignored if use_generator=True)
            charset: character set for random Generate (only used if use_generator=True)
            
        Returns: 
            list of example dictionaries with 'input' and 'output' keys
        """
        if self.use_generator:
            # Generate fresh random example on the fly
            return self.generate_examples(num_examples, charset=charset, seed=seed)
        else:
            # Use the standard BaseTask logic with stored data
            return super().get_icl_examples(
                num_examples=num_examples,
                shuffle=shuffle,
                seed=seed,
                fresh=fresh
            )
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions using exact-match accuracy."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) doesn't match "
                f"ground truth count ({len(ground_truth)})"
            )
        
        # Preprocess predictions
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate exact-match accuracy
        correct = sum(
            1 for pred, gt in zip(processed_predictions, ground_truth)
            if pred.strip() == gt.strip()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }


def make_copying_task(config: TaskConfig = None, 
                      use_generator: bool = False,
                      min_length: int = 3,
                      max_length: int = 8,
                      test_size: int = 20) -> CopyingTask:
    """factory function to Create a CopyingTask with default examples.
    
    Args: 
        configuration: optional TaskConfig. If None, creates a default configuration with
                example data where input == output.
        use_generator: If True, Generate random example on the fly instead of using static data.
                      This allows for limitless unique example.
        min_length: Minimum length of generated random string (only used if use_generator=True)
        max_length: Maximum length of generated random string (only used if use_generator=True)
        test_size: number of test examples to Generate(default: 20)
    
    Returns: 
        CopyingTask instance
    """
    if config is None:
        if use_generator:
            # Generate test example once at task creation for deterministic Evaluate
            # We Create a temporary instance just to use the generate_examples method
            import random
            import string
            
            # Generate a deterministic test set
            random.seed(42)
            test_examples = []
            for _ in range(test_size):
                length = random.randint(min_length, max_length)
                charset = string.ascii_letters + string.digits
                random_str = ''.join(random.choices(charset, k=length))
                test_examples.append({"input": random_str, "output": random_str})
        else:
            # default example - simple string that should be copied exactly
            test_examples = [
                {"input": "cat", "output": "cat"},
                {"input": "dog", "output": "dog"},
                {"input": "hello", "output": "hello"},
                {"input": "world", "output": "world"},
                {"input": "apple", "output": "apple"},
                {"input": "banana", "output": "banana"},
                {"input": "123", "output": "123"},
                {"input": "xyz", "output": "xyz"},
                {"input": "test", "output": "test"},
                {"input": "copy", "output": "copy"},
            ]
        
        config = TaskConfig(
            name="copying",
            description="Task where output exactly copies the input (induction heads)",
            data_format="memory",
            in_memory_data=test_examples,
            num_demonstrations=5,
        )
    
    return CopyingTask(config, use_generator=use_generator, 
                      min_length=min_length, max_length=max_length)
