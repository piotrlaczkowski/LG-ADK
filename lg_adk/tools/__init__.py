"""Tools for LG-ADK agents. Provides various tools for agent-environment interaction."""

from lg_adk.tools.agent_router import AgentRouter, RouterType
from lg_adk.tools.agent_tools import DelegationTool, MemoryTool, UserInfoTool, register_agent_tools
from lg_adk.tools.group_chat import GroupChatTool
from lg_adk.tools.retrieval import BaseRetrievalTool, ChromaDBRetrievalTool, Document, SimpleVectorRetrievalTool

try:
    from lg_adk.tools.morphik_retrieval import (
        MorphikGraphCreationTool,
        MorphikGraphTool,
        MorphikMCPTool,
        MorphikRetrievalTool,
    )
except ImportError:
    # Create dummy variables for type checking if Morphik is not installed
    MorphikRetrievalTool = None
    MorphikMCPTool = None
    MorphikGraphTool = None
    MorphikGraphCreationTool = None

__all__ = [
    "DelegationTool",
    "MemoryTool",
    "UserInfoTool",
    "register_agent_tools",
    "GroupChatTool",
    "AgentRouter",
    "RouterType",
    "BaseRetrievalTool",
    "SimpleVectorRetrievalTool",
    "ChromaDBRetrievalTool",
    "Document",
    "MorphikRetrievalTool",
    "MorphikMCPTool",
    "MorphikGraphTool",
    "MorphikGraphCreationTool",
]
