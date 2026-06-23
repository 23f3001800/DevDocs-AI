"""
Scripts package init — runs before any script module is imported.

Contains a compatibility shim for RAGAS: langchain-community >= 0.3.0
removed langchain_community.chat_models.vertexai, but RAGAS still
imports ChatVertexAI from it at module level. We stub both the module
AND the class so the import chain succeeds.
"""

import sys
import types

_VERTEXAI = "langchain_community.chat_models.vertexai"
if _VERTEXAI not in sys.modules:
    try:
        __import__(_VERTEXAI)
    except (ModuleNotFoundError, ImportError):
        _stub = types.ModuleType(_VERTEXAI)
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # dummy class
        sys.modules[_VERTEXAI] = _stub
elif not hasattr(sys.modules[_VERTEXAI], "ChatVertexAI"):
    sys.modules[_VERTEXAI].ChatVertexAI = type("ChatVertexAI", (), {})
