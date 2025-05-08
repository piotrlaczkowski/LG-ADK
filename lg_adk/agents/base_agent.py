from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool


class Agent:
    """Agent for LangGraph workflows using LangChain tools and prompts.

    Attributes:
        name: Name of the agent.
        llm: Language model instance.
        prompt: System prompt for the agent.
        tools: List of tools available to the agent.
    """

    def __init__(self, name: str, llm: object, prompt: str, tools: list[BaseTool]) -> None:
        """Initialize the Agent.

        Args:
            name: Name of the agent.
            llm: Language model instance.
            prompt: System prompt for the agent.
            tools: List of tools available to the agent.
        """
        self.name = name
        self.prompt = prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="messages"),
            ],
        )
        self.agent = prompt_template | llm.bind_tools(tools)
        self.tools = tools
