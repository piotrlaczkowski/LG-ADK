from langchain.tools import BaseTool

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool: BaseTool):
        self.tools[name] = tool

    def get_tool(self, name: str):
        return self.tools.get(name, None)

    def list_tools(self):
        return list(self.tools.keys())
 
