"""LG-ADK - LangGraph Agent Development Kit.

A development kit for building LangGraph-based agents with persistent memory,
session management, multi-agent capabilities, and local model support.
"""

__version__ = "0.1.0"

from lg_adk.agents.base import Agent
from lg_adk.agents.multi_agent import Conversation, MultiAgentSystem
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.config.settings import Settings
from lg_adk.eval.dataset import EvalDataset
from lg_adk.eval.evaluator import Evaluator
from lg_adk.memory.memory_manager import MemoryManager
from lg_adk.models import ModelRegistry, get_model
from lg_adk.sessions.session_manager import SessionManager

__all__ = [
    "Agent",
    "MultiAgentSystem",
    "Conversation",
    "GraphBuilder",
    "MemoryManager",
    "SessionManager",
    "Settings",
    "get_model",
    "ModelRegistry",
    "Evaluator",
    "EvalDataset",
]
