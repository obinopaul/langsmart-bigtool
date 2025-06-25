import inspect
import math
import types
import uuid
from typing import Callable

import pytest
from langchain_core.language_models import GenericFakeChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from langsmart_bigtool import create_agent
from langsmart_bigtool.graph import State, ToolSelectionResponse
from langsmart_bigtool.utils import convert_positional_only_function_to_tool


# Create a list of all the functions in the math module
all_names = dir(math)

math_functions = [
    getattr(math, name)
    for name in all_names
    if isinstance(getattr(math, name), types.BuiltinFunctionType)
]

# Convert to tools, handling positional-only arguments (idiosyncrasy of math module)
all_tools = []
for function in math_functions:
    if wrapper := convert_positional_only_function_to_tool(function):
        all_tools.append(wrapper)

# Store tool objects in registry
tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}


class FakeSelectorModel(GenericFakeChatModel):
    """Fake model that simulates tool selection."""
    
    def __init__(self, selected_tool_ids: list[str], **kwargs):
        self.selected_tool_ids = selected_tool_ids
        super().__init__(**kwargs)
    
    def with_structured_output(self, schema):
        def _invoke(messages):
            return ToolSelectionResponse(
                selected_tools=self.selected_tool_ids,
                reasoning="Selected tools based on query analysis"
            )
        
        class StructuredModel:
            def invoke(self, messages):
                return _invoke(messages)
            
            async def ainvoke(self, messages):
                return _invoke(messages)
        
        return StructuredModel()


class FakeMainModel(GenericFakeChatModel):
    """Fake model for the main agent."""
    
    def bind_tools(self, tools) -> "FakeMainModel":
        """Simulate binding tools."""
        self.bound_tools = tools
        return self


def _get_acos_tool_id():
    """Get the tool ID for the acos tool."""
    for tool_id, tool in tool_registry.items():
        if isinstance(tool, BaseTool) and tool.name == "acos":
            return tool_id
    raise ValueError("acos tool not found")


def test_llm_driven_tool_selection():
    """Test the new LLM-driven tool selection architecture."""
    acos_tool_id = _get_acos_tool_id()
    
    # Create fake models
    selector_llm = FakeSelectorModel(selected_tool_ids=[acos_tool_id])
    
    main_llm = FakeMainModel(
        messages=iter([
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "acos",
                        "args": {"x": 0.5},
                        "id": "abc234",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("The arc cosine of 0.5 is approximately 1.0472 radians."),
        ])
    )
    
    # Create agent with new architecture
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    # Test the agent
    result = agent.invoke({
        "messages": [HumanMessage(content="Use available tools to calculate arc cosine of 0.5.")]
    })
    
    # Validate results
    assert isinstance(result, dict)
    assert "messages" in result
    assert "selected_tool_ids" in result
    assert acos_tool_id in result["selected_tool_ids"]
    
    # Check that messages were processed
    messages = result["messages"]
    assert len(messages) >= 2  # Should have at least tool selection and final response
    
    # Check that the final message is from the AI
    final_message = messages[-1]
    assert isinstance(final_message, AIMessage)


def test_tool_selector_creates_manifest():
    """Test that the tool selector creates a proper tool manifest."""
    acos_tool_id = _get_acos_tool_id()
    
    selector_llm = FakeSelectorModel(selected_tool_ids=[acos_tool_id])
    main_llm = FakeMainModel(messages=iter([AIMessage("Test response")]))
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    # Test tool selection
    result = agent.invoke({
        "messages": [HumanMessage(content="Calculate arc cosine")]
    })
    
    assert acos_tool_id in result["selected_tool_ids"]


def test_main_agent_uses_selected_tools():
    """Test that the main agent only uses selected tools."""
    acos_tool_id = _get_acos_tool_id()
    
    selector_llm = FakeSelectorModel(selected_tool_ids=[acos_tool_id])
    
    # Create a main model that tracks tool binding
    bound_tools = []
    
    class TrackingMainModel(FakeMainModel):
        def bind_tools(self, tools):
            nonlocal bound_tools
            bound_tools = tools
            return super().bind_tools(tools)
    
    main_llm = TrackingMainModel(
        messages=iter([AIMessage("Calculated result")])
    )
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Calculate arc cosine of 0.5")]
    })
    
    # Verify that only selected tools were bound
    assert len(bound_tools) == 1
    assert bound_tools[0].name == "acos"


def test_empty_tool_selection():
    """Test behavior when no tools are selected."""
    selector_llm = FakeSelectorModel(selected_tool_ids=[])
    main_llm = FakeMainModel(messages=iter([AIMessage("No tools available")]))
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Do something")]
    })
    
    # Should handle empty tool selection gracefully
    assert "messages" in result


def test_invalid_tool_ids():
    """Test behavior when invalid tool IDs are selected."""
    selector_llm = FakeSelectorModel(selected_tool_ids=["invalid_id_1", "invalid_id_2"])
    main_llm = FakeMainModel(messages=iter([AIMessage("No valid tools")]))
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Do something")]
    })
    
    # Should handle invalid tool IDs gracefully
    assert "messages" in result
    # Invalid tool IDs should be filtered out
    assert result["selected_tool_ids"] == ["invalid_id_1", "invalid_id_2"]  # They're still in state, but filtered during execution


def test_multiple_tool_selection():
    """Test selection of multiple tools."""
    # Get multiple tool IDs
    tool_ids = []
    for tool_id, tool in tool_registry.items():
        if isinstance(tool, BaseTool) and tool.name in ["acos", "sin", "cos"]:
            tool_ids.append(tool_id)
            if len(tool_ids) >= 3:
                break
    
    selector_llm = FakeSelectorModel(selected_tool_ids=tool_ids)
    main_llm = FakeMainModel(messages=iter([AIMessage("Multiple tools available")]))
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Do trigonometry calculations")]
    })
    
    # Should have selected multiple tools
    assert len(result["selected_tool_ids"]) == len(tool_ids)
    for tool_id in tool_ids:
        assert tool_id in result["selected_tool_ids"]


@pytest.mark.asyncio
async def test_async_tool_selection():
    """Test async version of the tool selection."""
    acos_tool_id = _get_acos_tool_id()
    
    selector_llm = FakeSelectorModel(selected_tool_ids=[acos_tool_id])
    main_llm = FakeMainModel(
        messages=iter([AIMessage("Async calculation complete")])
    )
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Calculate arc cosine asynchronously")]
    })
    
    assert acos_tool_id in result["selected_tool_ids"]
    assert "messages" in result


def test_state_preservation():
    """Test that state is properly preserved between nodes."""
    acos_tool_id = _get_acos_tool_id()
    
    selector_llm = FakeSelectorModel(selected_tool_ids=[acos_tool_id])
    main_llm = FakeMainModel(
        messages=iter([AIMessage("State preserved")])
    )
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    
    initial_state = {
        "messages": [HumanMessage(content="Test state preservation")],
        "selected_tool_ids": []
    }
    
    result = agent.invoke(initial_state)
    
    # Original message should be preserved
    assert any(
        isinstance(msg, HumanMessage) and "Test state preservation" in msg.content
        for msg in result["messages"]
    )
    
    # Selected tools should be added
    assert acos_tool_id in result["selected_tool_ids"]