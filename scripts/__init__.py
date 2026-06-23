"""
Scripts package init — runs before any script module is imported.

Contains a compatibility shim for RAGAS: langchain-community >= 0.3.0
removed langchain_community.chat_models.vertexai, but RAGAS still
imports it at module level. We stub it here so the import succeeds.
"""

import sys
import types

_VERTEXAI = "langchain_community.chat_models.vertexai"
if _VERTEXAI not in sys.modules:
    try:
        __import__(_VERTEXAI)
    except ModuleNotFoundError:
        sys.modules[_VERTEXAI] = types.ModuleType(_VERTEXAI)
