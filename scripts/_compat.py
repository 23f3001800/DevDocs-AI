"""
Compatibility shim for RAGAS + langchain-community >= 0.3.0.

RAGAS imports langchain_community.chat_models.vertexai which was
removed in langchain-community >= 0.3.0. We don't use VertexAI
(we use Google GenAI), so we create a stub module to prevent
the ImportError at ragas startup.

Import this module BEFORE importing ragas.
"""

import sys
import types

_vertexai_module = "langchain_community.chat_models.vertexai"
if _vertexai_module not in sys.modules:
    try:
        __import__(_vertexai_module)
    except ModuleNotFoundError:
        sys.modules[_vertexai_module] = types.ModuleType(_vertexai_module)
