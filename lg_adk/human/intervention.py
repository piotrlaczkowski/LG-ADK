"""
Human intervention for LangGraph workflows.
"""

import asyncio
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, Field


class HumanIntervention(BaseModel):
    """
    Handles human intervention in LangGraph workflows.
    
    Attributes:
        input_callback: Callback function to get input from a human.
        auto_approve: Whether to automatically approve without human input.
        timeout: Timeout in seconds for human input.
    """
    
    input_callback: Optional[Callable[[Dict[str, Any]], str]] = Field(
        None, description="Callback to get input from a human"
    )
    auto_approve: bool = Field(False, description="Auto-approve without human input")
    timeout: int = Field(300, description="Timeout in seconds for human input")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state with human intervention.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            The updated state after human intervention.
        """
        # If auto_approve is enabled, just pass through
        if self.auto_approve:
            return {**state, "human_input": "Auto-approved"}
        
        # Display the current state to the human
        agent = state.get("agent", "Unknown agent")
        output = state.get("output", "")
        
        # Get human input
        human_input = "Approved"
        
        if self.input_callback:
            prompt = (
                f"Agent '{agent}' produced the following output:\n\n"
                f"{output}\n\n"
                f"Enter 'Approved' to approve, or provide feedback:"
            )
            
            # This would call the callback to get input in a real implementation
            human_input = self.input_callback({"prompt": prompt, "state": state})
        else:
            # Default implementation for console input
            print(f"Agent '{agent}' produced the following output:")
            print(f"{output}")
            human_input = input("Enter 'Approved' to approve, or provide feedback: ")
        
        # Update the state with human input
        return {**state, "human_input": human_input}
    
    async def async_intervene(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state with human intervention asynchronously.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            The updated state after human intervention.
        """
        # Run the synchronous method in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.__call__, state) 