"""
function vector Extract package for ICL example.

This package contains utilities for extracting function vectors from 
in-context learning example across different task.
"""

from .extract_function_vecs import discover_all_tasks

__all__ = ['discover_all_tasks']