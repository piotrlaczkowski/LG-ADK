"""
Base Agent class for LG-ADK.
"""

from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import BaseTool
from langgraph.graph import Graph, StateGraph
from pydantic import BaseModel, Field

from lg_adk.models import get_model


class Agent(BaseModel):
    """
    Base Agent class that represents an agent in a LangGraph workflow.
    
    Attributes:
        name: The name of the agent.
        llm: The language model to use (can be a provider-prefixed name like "ollama/llama3", "gemini/gemini-pro").
        description: A description of the agent's purpose.
        tools: A list of tools the agent can use.
        system_message: System message to provide context to the agent.
    """
    
    name: str = Field(..., description="Name of the agent")
    llm: Any = Field(None, description="Language model to use")
    description: str = Field("A helpful AI assistant", description="Description of the agent's purpose")
    tools: List[BaseTool] = Field(default_factory=list, description="Tools available to the agent")
    system_message: str = Field(
        "", 
        description="System message to provide context to the agent"
    )
    _model: Any = None
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    def add_tools(self, tools: List[BaseTool]) -> None:
        """Add multiple tools to the agent."""
        self.tools.extend(tools)
    
    def create_prompt(self) -> str:
        """Create a prompt for the agent based on its configuration."""
        if self.system_message:
            return self.system_message
        
        prompt = f"You are {self.name}, {self.description}.\n\n"
        
        if self.tools:
            prompt += "You have access to the following tools:\n\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
        
        return prompt
    
    def get_model(self) -> Any:
        """Get the underlying language model."""
        if self._model is None:
            if isinstance(self.llm, str):
                self._model = get_model(self.llm)
            else:
                # If it's already a model instance, use it directly
                self._model = self.llm
                
        return self._model
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return a new state.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            The updated state after the agent's processing.
        """
        user_input = state.get("input", "")
        
        # Create the prompt
        prompt = self.create_prompt()
        prompt += f"\n\nUser: {user_input}"
        
        # Get the model and generate a response
        model = self.get_model()
        
        try:
            response = model.invoke(prompt)
        except Exception as e:
            # If we encounter an error, return a placeholder response for now
            # In a real implementation, we would handle this more gracefully
            response = f"Error generating response: {str(e)}"
        
        return {"output": response, "agent": self.name}
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the agent callable directly."""
        return self.run(state) 