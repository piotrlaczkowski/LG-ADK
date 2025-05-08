from langchain.tools import BaseTool


class ToolManager:
    """Manages registration and retrieval of tools for agents.

    Attributes:
        tools: Dictionary mapping tool names to tool instances.
    """

    def __init__(self) -> None:
        """Initialize the ToolManager with an empty tool registry."""
        self.tools: dict[str, BaseTool] = {}

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool with a given name.

        Args:
            name: Name of the tool.
            tool: Tool instance to register.
        """
        self.tools[name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool instance if found, else None.
        """
        return self.tools.get(name, None)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self.tools.keys())
