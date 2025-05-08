"""Web search tool for agents."""

from langchain.schema import BaseTool
from pydantic import Field


class WebSearchTool(BaseTool):
    """Tool for performing web searches.

    Attributes:
        name: Name of the tool.
        description: Description of the tool.
        search_engine: Search engine to use.
    """

    name: str = "web_search"
    description: str = "Search the web for information on a topic or question."
    search_engine: str = Field("duckduckgo", description="Search engine to use")

    def _run(self, query: str) -> str:
        """Run the tool with the given query.

        Args:
            query: The search query.

        Returns:
            The search results as a string.
        """
        # In a real implementation, this would use an actual search API
        # For this example, we'll just return a placeholder response
        return (
            f"Results for '{query}':\n"
            "1. Example search result 1\n"
            "2. Example search result 2\n"
            "3. Example search result 3"
        )

    async def _arun(self, query: str) -> str:
        """Run the tool asynchronously with the given query.

        Args:
            query: The search query.

        Returns:
            The search results as a string.
        """
        # This would use an async API in a real implementation
        return self._run(query)
