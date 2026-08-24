"""Ignoring context task - Information retrieval with distractors.

Tests the model's ability to Extract relevant information while ignoring
irrelevant context. Similar to "needle in a haystack" but simpler.

Format: 
    [Filler text] KEY FACT [More filler] Question: [Query about KEY FACT]
    Answer: [Extracted information]

Example: 
    Some text here. X = 5. More text here.
    Question: What is X?
    Answer: 5

This tests selective attention and information retrieval capabilities.
"""

import random
import string
from typing import Dict, List, Optional
from tasks.base_task import BaseTask, TaskConfig


class IgnoringContextTask(BaseTask):
    """
    task testing the ability to Extract key information from irrelevant context.
    
    The model must find and Extract a variable assignment buried in filler text.
    """
    TASK_NAME = "ignoring_context"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False):
        """
        Initialize IgnoringContextTask.
        
        Args: 
            configuration: a TaskConfig with task settings
            use_generator: If True, Generate synthetic example with random variable
        """
        self.use_generator = use_generator
        super().__init__(config)
    
    def _generate_filler_text(self, num_words: int = 5, seed: Optional[int] = None) -> str:
        """Generate random filler text."""
        if seed is not None:
            random.seed(seed)
        
        # simple word for filler
        filler_words = [
            "the", "a", "is", "was", "are", "were", "has", "have", "had",
            "today", "yesterday", "tomorrow", "here", "there", "then", "now",
            "cat", "dog", "tree", "house", "car", "book", "day", "time",
            "good", "bad", "nice", "big", "small", "red", "blue", "green",
            "went", "came", "saw", "said", "made", "took", "gave", "found"
        ]
        
        words = [random.choice(filler_words) for _ in range(num_words)]
        return " ".join(words).capitalize() + "."
    
    def generate_example(
        self,
        variable: Optional[str] = None,
        value: Optional[int] = None,
        filler_before: int = 5,
        filler_after: int = 5,
        seed: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate a single example with a variable assignment buried in context.
        
        Args: 
            variable: variable name (e.g., "X"). If None, randomly chosen.
            value: value to assign. If None, random integer 1-20.
            filler_before: number of filler words before the key fact.
            filler_after: number of filler words after the key fact.
            seed: random seed for reproducibility.
            
        Returns: 
            dictionary with input, output, variable, and value.
        """
        if seed is not None:
            random.seed(seed)
        
        # Generate variable and value if not provided
        if variable is None:
            variable = random.choice(['X', 'Y', 'Z', 'A', 'B', 'N', 'M'])
        
        if value is None:
            value = random.randint(1, 20)
        
        # Generate filler text
        before = self._generate_filler_text(filler_before, seed)
        after = self._generate_filler_text(filler_after, seed=(seed+1) if seed else None)
        
        # Build the context with embedded fact
        context = f"{before} {variable} = {value}. {after}"
        question = f"Question: What is {variable}?"
        
        # Combined input
        input_str = f"{context}\n{question}"
        
        return {
            "input": input_str,
            "output": str(value),
            "variable": variable,
            "value": value,
            "context_length": len(context.split())
        }
    
    def generate_examples(
        self,
        num_examples: int,
        filler_range: tuple = (3, 8),
        seed: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Generate multiple example with varying context lengths.
        
        Args: 
            num_examples: number of examples to Generate.
            filler_range: (min, max) word for filler text.
            seed: random seed for reproducibility.
            
        Returns: 
            list of example dictionaries.
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for i in range(num_examples):
            filler_len = random.randint(*filler_range)
            example = self.generate_example(
                filler_before=filler_len,
                filler_after=filler_len,
                seed=(seed + i) if seed else None
            )
            examples.append(example)
        
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
        filler_range: tuple = (3, 8),
    ) -> List[Dict[str, str]]:
        """
        Return examples in ICL format.
        
        If use_generator=True, Generate synthetic example.
        Otherwise, uses the standard BaseTask logic with stored data.
        
        Args: 
            num_examples: number of examples to return.
            shuffle: Whether to shuffle (ignored if use_generator=True).
            seed: random seed for reproducibility.
            fresh: Whether to prefer unused example (ignored if use_generator=True).
            filler_range: (min, max) word for filler text in generated example.
            
        Returns: 
            list of example dictionaries with 'input' and 'output' keys.
        """
        if self.use_generator:
            # Generate fresh example on the fly
            return self.generate_examples(num_examples, filler_range, seed)
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


def make_ignoring_context_task(
    config: TaskConfig = None,
    use_generator: bool = False,
) -> IgnoringContextTask:
    """factory function to Create a IgnoringContextTask.
    
    Args: 
        configuration: optional TaskConfig. If None, creates default configuration with example.
        use_generator: If True, Generate synthetic example on the fly.
    
    Returns: 
        IgnoringContextTask instance.
    """
    if config is None:
        if use_generator:
            # Minimal configuration for generator mode
            default_examples = [
                {
                    "input": "Some text here. X = 5. More text.\nQuestion: What is X?",
                    "output": "5"
                }
            ]
        else:
            # static example with varying context lengths
            default_examples = [
                {
                    "input": "The cat sat. X = 3. A dog ran.\nQuestion: What is X?",
                    "output": "3",
                    "variable": "X",
                    "value": 3
                },
                {
                    "input": "Today was nice and sunny. Y = 7. We went to the park.\nQuestion: What is Y?",
                    "output": "7",
                    "variable": "Y",
                    "value": 7
                },
                {
                    "input": "The house is big and blue. Z = 12. Birds were singing.\nQuestion: What is Z?",
                    "output": "12",
                    "variable": "Z",
                    "value": 12
                },
                {
                    "input": "A car drove by fast. N = 1. The tree is tall.\nQuestion: What is N?",
                    "output": "1",
                    "variable": "N",
                    "value": 1
                },
                {
                    "input": "Books are good to read every day. M = 15. Time goes quickly.\nQuestion: What is M?",
                    "output": "15",
                    "variable": "M",
                    "value": 15
                },
            ]
        
        config = TaskConfig(
            name="ignoring_context",
            description="Extract key information while ignoring irrelevant context",
            data_format="memory",
            in_memory_data=default_examples,
            num_demonstrations=3,
        )
    
    return IgnoringContextTask(config, use_generator=use_generator)
