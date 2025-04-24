from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage
from typing import List

class Agent:
    def __init__(self, name: str, llm, prompt: str, tools: List[BaseTool]):
        self.name = name
        self.prompt = prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.agent = prompt_template | llm.bind_tools(tools)
        self.tools = tools
