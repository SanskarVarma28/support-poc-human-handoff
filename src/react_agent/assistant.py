from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from pydantic import BaseModel, Field

from react_agent.state import State
from react_agent.utils import load_chat_model
from react_agent.tools.lookup_knowledge_base import lookup_knowledge_base

llm = load_chat_model("anthropic/claude-3-5-sonnet-20241022")

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


class ToHumanAssistant(BaseModel):
    """Transfers work to a human assistant to handle resolve customer issues, requests and queries."""

    email: str = Field(
        description="The email id of the customer."
    )
    request: str = Field(
        description="Summary of the customer issue/request/query"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.bale@outlook.com",
                "request": "The user wants to know how to disconnect his gmail from supersales.",
            }
        }


# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for SuperAGI. "
            "Your primary role is to lookup knowledge base to answer customer queries regarding SuperAGI's product, pricing, technology, etc. "
            "If a customer makes any request which you are unable to handle, or asks any query which you are unable to answer, "
            "delegate the task to the human assistant by invoking the corresponding tool."
            # " You are not able to book demo calls yourself."
            # " Only the specialized assistants are given permission to do this for the user."
            # "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            # "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            # " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            # " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user information:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)
primary_assistant_tools = [
    lookup_knowledge_base,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToHumanAssistant
    ]
)