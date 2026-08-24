"""
function vector 擷取 package for ICL 範例.

This package contains utilities for extracting function vectors from 
in-context learning 範例 across different 任務.
"""

from .extract_function_vecs import discover_all_tasks

__all__ = ['discover_all_tasks']