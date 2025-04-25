# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI()
# llm.invoke("Hello, world!")

from typing import Annotated, Literal
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from library.setup_tools import tavily_tool, python_repl_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

memory = MemorySaver()

members = ["workout_plan_generator", "diet_plan_generator"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options] # type: ignore


# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
llm = ChatOpenAI(model="gpt-4o")


class State(TypedDict):
    next: str
    messages: Annotated[list, add_messages]


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]: # type: ignore
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


workout_plan_generator_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt="You are a workout plan generator. Based on user input and fitness goals, please generate a workout plan"
)


def workout_plan_generator_node(state: State) -> Command[Literal["supervisor"]]:
    result = workout_plan_generator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
diet_plan_generator_agent = create_react_agent(llm, tools=[tavily_tool], prompt="You are a diet plan generator. Based on user input and fitness goals, please generate a diet plan")


def diet_plan_generator_node(state: State) -> Command[Literal["supervisor"]]:
    result = diet_plan_generator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("workout_plan_generator", workout_plan_generator_agent)
builder.add_node("diet_plan_generator", diet_plan_generator_agent)
graph = builder.compile(checkpointer=memory)

# for s in graph.stream(
#     {"messages": [("user", "I want to get in shape. I want to eat healthy and usually that includes indian food. Create a diet plan for a the next 3 months")]}, subgraphs=True
# ):
#     print(s)
#     print("----")