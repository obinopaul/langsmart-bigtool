from typing import Annotated, Callable, List
import json

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field


def _add_new(left: list, right: list) -> list:
    """Extend left list with new items from right list."""
    return left + [item for item in right if item not in set(left)]


class State(MessagesState):
    selected_tool_ids: Annotated[list[str], _add_new]


class ToolSelectionResponse(BaseModel):
    """Structured response for tool selection."""
    selected_tools: List[str] = Field(
        description="List of tool names selected for the task"
    )
    reasoning: str = Field(
        description="Brief explanation of why these tools were selected"
    )


def create_agent(
    selector_llm: LanguageModelLike,
    main_llm: LanguageModelLike,
    tool_registry: dict[str, BaseTool | Callable],
) -> StateGraph:
    """Create an agent with LLM-driven tool selection.

    The agent uses a two-node architecture:
    1. Tool Selector: A fast LLM that selects relevant tools from the registry
    2. Main Agent: A ReAct agent that uses only the selected tools

    Args:
        selector_llm: Fast language model for tool selection.
        main_llm: Language model for the main agent execution.
        tool_registry: Dict mapping string IDs to tools or callables.
    """
    
    def _create_tool_manifest(tool_registry: dict[str, BaseTool | Callable]) -> str:
        """Create a structured manifest of all available tools."""
        manifest_lines = ["# Available Tools\n"]
        
        for tool_id, tool in tool_registry.items():
            if isinstance(tool, BaseTool):
                name = tool.name
                description = tool.description
            else:
                name = tool.__name__
                description = tool.__doc__ or "No description available"
            
            manifest_lines.append(f"## {name}")
            manifest_lines.append(f"- **ID**: {tool_id}")
            manifest_lines.append(f"- **Description**: {description}")
            manifest_lines.append("")
        
        return "\n".join(manifest_lines)

    def tool_selector(state: State, config: RunnableConfig) -> State:
        """Select relevant tools based on the user's query."""
        messages = state["messages"]
        user_query = messages[-1].content if messages else ""
        
        tool_manifest = _create_tool_manifest(tool_registry)
        
        system_prompt = f"""You are a tool selection expert. Your task is to analyze a user's query and select the most relevant tools from the available tool registry.

{tool_manifest}

Instructions:
1. Analyze the user's query carefully
2. Select 3-10 of the most relevant tools that could help answer the query
3. Provide the tool IDs (not names) in your response
4. Explain your reasoning briefly

Return your response as a JSON object with the following structure:
{{
    "selected_tools": ["tool_id_1", "tool_id_2", ...],
    "reasoning": "Brief explanation of why these tools were selected"
}}

User Query: {user_query}"""

        selector_with_structured_output = selector_llm.with_structured_output(
            ToolSelectionResponse
        )
        
        response = selector_with_structured_output.invoke([
            SystemMessage(content=system_prompt)
        ])
        
        return {
            "selected_tool_ids": response.selected_tools,
            "messages": [
                HumanMessage(
                    content=f"Selected tools: {response.selected_tools}. Reasoning: {response.reasoning}"
                )
            ]
        }

    async def atool_selector(state: State, config: RunnableConfig) -> State:
        """Async version of tool selector."""
        messages = state["messages"]
        user_query = messages[-1].content if messages else ""
        
        tool_manifest = _create_tool_manifest(tool_registry)
        
        system_prompt = f"""You are a tool selection expert. Your task is to analyze a user's query and select the most relevant tools from the available tool registry.

{tool_manifest}

Instructions:
1. Analyze the user's query carefully
2. Select 3-10 of the most relevant tools that could help answer the query
3. Provide the tool IDs (not names) in your response
4. Explain your reasoning briefly

Return your response as a JSON object with the following structure:
{{
    "selected_tools": ["tool_id_1", "tool_id_2", ...],
    "reasoning": "Brief explanation of why these tools were selected"
}}

User Query: {user_query}"""

        selector_with_structured_output = selector_llm.with_structured_output(
            ToolSelectionResponse
        )
        
        response = await selector_with_structured_output.ainvoke([
            SystemMessage(content=system_prompt)
        ])
        
        return {
            "selected_tool_ids": response.selected_tools,
            "messages": [
                HumanMessage(
                    content=f"Selected tools: {response.selected_tools}. Reasoning: {response.reasoning}"
                )
            ]
        }

    def main_agent(state: State, config: RunnableConfig) -> State:
        """Main agent that uses only the selected tools."""
        selected_tools = [
            tool_registry[tool_id] for tool_id in state["selected_tool_ids"]
            if tool_id in tool_registry
        ]
        
        if not selected_tools:
            return {
                "messages": [
                    AIMessage(content="No valid tools were selected. Please try rephrasing your query.")
                ]
            }
        
        llm_with_tools = main_llm.bind_tools(selected_tools)
        
        # Get the original user query (first message)
        original_query = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Selected tools:"):
                original_query = msg.content
                break
        
        if original_query:
            # Create a fresh conversation with the original query
            conversation = [HumanMessage(content=original_query)]
        else:
            conversation = state["messages"]
        
        response = llm_with_tools.invoke(conversation)
        return {"messages": [response]}

    async def amain_agent(state: State, config: RunnableConfig) -> State:
        """Async version of main agent."""
        selected_tools = [
            tool_registry[tool_id] for tool_id in state["selected_tool_ids"]
            if tool_id in tool_registry
        ]
        
        if not selected_tools:
            return {
                "messages": [
                    AIMessage(content="No valid tools were selected. Please try rephrasing your query.")
                ]
            }
        
        llm_with_tools = main_llm.bind_tools(selected_tools)
        
        # Get the original user query (first message)
        original_query = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Selected tools:"):
                original_query = msg.content
                break
        
        if original_query:
            # Create a fresh conversation with the original query
            conversation = [HumanMessage(content=original_query)]
        else:
            conversation = state["messages"]
        
        response = await llm_with_tools.ainvoke(conversation)
        return {"messages": [response]}

    # Create tool node with all tools for execution
    tool_node = ToolNode([tool for tool in tool_registry.values() if isinstance(tool, BaseTool)])

    def should_continue(state: State) -> str:
        """Determine if the agent should continue, call tools, or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        else:
            return "tools"

    # Build the graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("tool_selector", tool_selector)
    builder.add_node("main_agent", main_agent)
    builder.add_node("tools", tool_node)
    
    # Set entry point
    builder.set_entry_point("tool_selector")
    
    # Add edges
    builder.add_edge("tool_selector", "main_agent")
    builder.add_conditional_edges(
        "main_agent",
        should_continue,
        path_map={"tools": "tools", END: END},
    )
    builder.add_edge("tools", "main_agent")
    
    return builder