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

# Try to import TrustCall, fallback to structured output if not available
try:
    from trustcall import create_extractor
    TRUSTCALL_AVAILABLE = True
except ImportError:
    TRUSTCALL_AVAILABLE = False


def _add_new(left: list, right: list) -> list:
    """Extend left list with new items from right list."""
    return left + [item for item in right if item not in set(left)]


class State(MessagesState):
    selected_tool_ids: Annotated[list[str], _add_new]


class ToolSelectionResponse(BaseModel):
    """Structured response for tool selection."""
    tool_ids: List[str] = Field(
        description="List of tool IDs selected for the task"
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

    if TRUSTCALL_AVAILABLE:
        # Use TrustCall for robust structured output
        tool_selector_extractor = create_extractor(
            selector_llm,
            tools=[ToolSelectionResponse],
            tool_choice="ToolSelectionResponse"
        )
        
        def _invoke_tool_selector(system_prompt: str) -> ToolSelectionResponse:
            result = tool_selector_extractor.invoke({
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Select the most relevant tools for this query:\n{system_prompt}"
                    }
                ]
            })
            return result["responses"][0]
    else:
        # Fallback to standard structured output
        selector_with_structured_output = selector_llm.with_structured_output(
            ToolSelectionResponse
        )
        
        def _invoke_tool_selector(system_prompt: str) -> ToolSelectionResponse:
            return selector_with_structured_output.invoke([
                SystemMessage(content=system_prompt)
            ])

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
4. If no tools are relevant to the query, return an empty list
5. Explain your reasoning briefly

User Query: {user_query}"""

        tool_selection = _invoke_tool_selector(system_prompt)
        
        # CRITICAL FIX: Append to messages instead of replacing
        return {
            "selected_tool_ids": tool_selection.tool_ids,
            "messages": [
                AIMessage(
                    content=f"Selected tools: {tool_selection.tool_ids}. Reasoning: {tool_selection.reasoning}"
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
4. If no tools are relevant to the query, return an empty list
5. Explain your reasoning briefly

User Query: {user_query}"""

        if TRUSTCALL_AVAILABLE:
            # TrustCall doesn't have async invoke yet, so we use sync for now
            # This can be updated when async support is available
            tool_selection = _invoke_tool_selector(system_prompt)
        else:
            # Use async structured output
            selector_with_structured_output = selector_llm.with_structured_output(
                ToolSelectionResponse
            )
            tool_selection = await selector_with_structured_output.ainvoke([
                SystemMessage(content=system_prompt)
            ])
        
        # CRITICAL FIX: Append to messages instead of replacing
        return {
            "selected_tool_ids": tool_selection.tool_ids,
            "messages": [
                AIMessage(
                    content=f"Selected tools: {tool_selection.tool_ids}. Reasoning: {tool_selection.reasoning}"
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
        
        # CRITICAL FIX: Use entire conversation history instead of string matching
        response = llm_with_tools.invoke(state["messages"])
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
        
        # CRITICAL FIX: Use entire conversation history instead of string matching
        response = await llm_with_tools.ainvoke(state["messages"])
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