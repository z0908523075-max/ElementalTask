"""
Test the simplified one-stop interface for function vector 擷取.
"""
import pytest
import torch
import os
import numpy as np

from function_vecs.extract_function_vecs import extract_function_vector_simple
from tasks.base_task import TaskConfig


@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ and os.environ.get("CI") == "true",
    reason="Skip on CI by default to avoid network downloads."
)
def test_simple_interface_existing_task():
    """Test the simplified interface with an existing 任務."""
    print("Testing simplified interface with existing task...")
    
    # Use one of our working basic ICL 任務
    function_vec = extract_function_vector_simple(
        task_name="basic_arithmetic",
        model_name="distilgpt2",  # Smaller 模型 for testing
        num_samples=3,
        device="cpu"  # Force CPU to avoid CUDA issues
    )

    # Verify the 結果
    assert function_vec.task_name == "basic_arithmetic"
    assert function_vec.function_vec.shape[0] > 0  # Should have some dimensions
    assert abs(function_vec.function_vec.dot(function_vec.function_vec) - 1.0) < 1e-5  # Should be L2 normalized
    
    print(f"✅ Existing task test passed!")
    print(f"   Task: {function_vec.task_name}")
    print(f"   Shape: {function_vec.function_vec.shape}")
    print(f"   Norm: {function_vec.function_vec.dot(function_vec.function_vec):.6f}")


@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ and os.environ.get("CI") == "true",
    reason="Skip on CI by default to avoid network downloads."
)
def test_simple_interface_different_tasks():
    """Test the simplified interface with different built-in 任務."""
    print("Testing simplified interface with different tasks...")
    
    # Test multiple basic ICL 任務 (use actual 可用 任務)
    tasks_to_test = ["simple_icl", "token_reversal", "part_of_speech"]
    
    for task_name in tasks_to_test:
        print(f"Testing task: {task_name}")
        function_vec = extract_function_vector_simple(
            task_name=task_name,
            model_name="distilgpt2",
            num_samples=3,
            device="cpu"
        )
        
        # Verify the 結果
        assert function_vec.task_name == task_name
        assert function_vec.function_vec.shape[0] > 0
        assert abs(function_vec.function_vec.dot(function_vec.function_vec) - 1.0) < 1e-5
        
        print(f"   ✅ {task_name}: shape {function_vec.function_vec.shape}")
    
    print(f"✅ Different tasks test passed!")


@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ and os.environ.get("CI") == "true",
    reason="Skip on CI by default to avoid network downloads."
)
def test_simple_interface_defaults():
    """Test that the simplified interface works with all 預設."""
    print("Testing simplified interface with defaults...")
    
    # Test with minimal parameters - should use all 預設
    function_vec = extract_function_vector_simple("basic_arithmetic")

    # Verify basic properties
    assert function_vec.task_name == "basic_arithmetic"
    assert function_vec.function_vec.shape[0] > 0
    assert abs(function_vec.function_vec.dot(function_vec.function_vec) - 1.0) < 1e-5
    
    print(f"✅ Defaults test passed!")
    print(f"   Used all default parameters successfully")


@pytest.mark.integration
def test_readme_example():
    """Test the README 範例 works as advertised."""
    
    print("Testing simple interface README example...")
    
    # 簡單 usage - use a working basic ICL 任務
    function_vec = extract_function_vector_simple("basic_arithmetic", num_samples=3)
    print(f"Function vector shape: {function_vec.function_vec.shape}")

    # Verify it worked
    assert function_vec.function_vec.shape[0] > 0
    assert abs(np.linalg.norm(function_vec.function_vec) - 1.0) < 1e-6  # Should be L2 normalized
    assert function_vec.task_name == "basic_arithmetic"
    
    print("✅ README example test passed!")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_simple_interface_existing_task()
    test_simple_interface_different_tasks() 
    test_simple_interface_defaults()
    test_readme_example()
    print("\n🎉 All simplified interface tests passed!")
