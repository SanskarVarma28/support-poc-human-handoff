import json
from typing import Callable, Literal

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,  END, START
from langgraph.prebuilt import tools_condition

from react_agent.assistant import *
from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.utils import create_tool_node_with_fallback, get_message_text

builder = StateGraph(State, input=InputState, config_schema=Configuration)

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate tool to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def user_info(state: State, config: RunnableConfig) -> State:
    configurable = Configuration.from_runnable_config(config)
    user_info = {"email":configurable.email, "name":configurable.name, "account_id":configurable.account_id}
    return {"user_info": json.dumps(user_info)}


def human_assistant(state: State, config: RunnableConfig) -> State:
    if get_message_text(state["messages"][-1]).__contains__("no thanks"):
        return {
            "messages": [
                AIMessage(
                    content= "Bye, Have a great day!",
                    tool_calls= [
                        {
                        "name": "CompleteOrEscalate",
                        "args": { "cancel": False },
                        "id": "tool_call_id",
                        "type": "tool_call",
                        },
                    ]
                ),
            ]
        }
    else:
        return {"messages": [AIMessage("How can I help you")]}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# Flight booking assistant
builder.add_node(
    "enter_human_assistant",
    create_entry_node("Human Assistant", "human_assistant"),
)

builder.add_node("human_assistant", human_assistant)
builder.add_edge("enter_human_assistant", "human_assistant")

# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)

def route_from_human(
    state: State,
) -> Literal[
    "leave_skill",
    "__end__",
]:
    if state["messages"][-1].tool_calls:
        return "leave_skill"
    else: 
        return "__end__"

builder.add_edge("leave_skill", "primary_assistant")
builder.add_conditional_edges("human_assistant", route_from_human)

# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToHumanAssistant.__name__:
            return "enter_human_assistant"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_human_assistant",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "human_assistant",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

support_agent_graph = builder.compile()