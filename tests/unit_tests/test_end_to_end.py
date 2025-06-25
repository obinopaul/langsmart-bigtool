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


class MockTrustCallExtractor:
    """Mock TrustCall extractor for testing."""
    
    def __init__(self, selected_tool_ids: list[str], reasoning: str = "Mock reasoning"):
        self.selected_tool_ids = selected_tool_ids
        self.reasoning = reasoning
    
    def invoke(self, input_data):
        """Mock invoke method that returns structured response."""
        return {
            "responses": [
                ToolSelectionResponse(
                    tool_ids=self.selected_tool_ids,
                    reasoning=self.reasoning
                )
            ]
        }


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


def test_improved_state_management():
    """Test that the improved state management preserves message history."""
    acos_tool_id = _get_acos_tool_id()
    
    # Mock the TrustCall extractor
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor([acos_tool_id], "Selected acos for arc cosine calculation")
    
    # Patch the create_extractor function
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(
            messages=iter([AIMessage("Calculation complete")])
        )
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        # Test with initial user message
        initial_query = "Calculate the arc cosine of 0.5"
        result = agent.invoke({
            "messages": [HumanMessage(content=initial_query)]
        })
        
        # Verify that original user message is preserved
        messages = result["messages"]
        assert any(
            isinstance(msg, HumanMessage) and initial_query in msg.content
            for msg in messages
        ), "Original user message should be preserved"
        
        # Verify tool selection message was added
        assert any(
            isinstance(msg, AIMessage) and "Selected tools:" in msg.content
            for msg in messages
        ), "Tool selection message should be added"
        
        # Verify selected tools
        assert acos_tool_id in result["selected_tool_ids"]
        
    finally:
        # Restore original function
        graph_module.create_extractor = original_create_extractor


def test_trustcall_integration():
    """Test TrustCall integration for robust tool selection."""
    acos_tool_id = _get_acos_tool_id()
    
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor(
            [acos_tool_id], 
            "Selected acos tool based on query analysis for arc cosine calculation"
        )
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(messages=iter([AIMessage("TrustCall test complete")]))
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        result = agent.invoke({
            "messages": [HumanMessage(content="Test TrustCall functionality")]
        })
        
        # Verify TrustCall structured output
        assert acos_tool_id in result["selected_tool_ids"]
        
        # Check reasoning was included
        reasoning_messages = [
            msg for msg in result["messages"] 
            if isinstance(msg, AIMessage) and "Selected tools:" in msg.content
        ]
        assert len(reasoning_messages) > 0
        assert "query analysis" in reasoning_messages[0].content
        
    finally:
        graph_module.create_extractor = original_create_extractor


def test_main_agent_uses_full_conversation():
    """Test that main agent uses full conversation history."""
    acos_tool_id = _get_acos_tool_id()
    
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor([acos_tool_id])
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        # Track what messages the main LLM receives
        received_messages = []
        
        class TrackingMainModel(FakeMainModel):
            def invoke(self, messages):
                nonlocal received_messages
                received_messages = messages
                return super().invoke(messages)
            
            def bind_tools(self, tools):
                self.bound_tools = tools
                return self
        
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = TrackingMainModel(
            messages=iter([AIMessage("Full conversation preserved")])
        )
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        # Multi-turn conversation
        initial_messages = [
            HumanMessage(content="Hello, I need help with math"),
            AIMessage(content="I can help with math calculations"),
            HumanMessage(content="Calculate arc cosine of 0.5")
        ]
        
        result = agent.invoke({"messages": initial_messages})
        
        # Verify main agent received full conversation history
        assert len(received_messages) >= 3, "Main agent should receive full conversation history"
        
        # Check that original messages are included
        original_contents = [msg.content for msg in initial_messages]
        received_contents = [msg.content for msg in received_messages]
        
        for original_content in original_contents:
            assert any(original_content in received_content for received_content in received_contents), \
                f"Original message '{original_content}' should be preserved"
        
    finally:
        graph_module.create_extractor = original_create_extractor


def test_no_valid_tools_scenario():
    """Test behavior when no valid tools are selected."""
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        # Return invalid tool IDs
        return MockTrustCallExtractor(["invalid_id_1", "invalid_id_2"], "No valid tools found")
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(messages=iter([AIMessage("No tools available")]))
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        result = agent.invoke({
            "messages": [HumanMessage(content="Do something impossible")]
        })
        
        # Should handle gracefully
        assert "messages" in result
        
        # Check for appropriate error message
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert "No valid tools were selected" in final_message.content
        
    finally:
        graph_module.create_extractor = original_create_extractor


def test_multiple_tool_selection_with_trustcall():
    """Test selection of multiple tools using TrustCall."""
    # Get multiple tool IDs
    tool_ids = []
    for tool_id, tool in tool_registry.items():
        if isinstance(tool, BaseTool) and tool.name in ["acos", "sin", "cos"]:
            tool_ids.append(tool_id)
            if len(tool_ids) >= 3:
                break
    
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor(
            tool_ids, 
            "Selected trigonometric functions for comprehensive math support"
        )
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(messages=iter([AIMessage("Multiple tools ready")]))
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        result = agent.invoke({
            "messages": [HumanMessage(content="Help with trigonometry")]
        })
        
        # Should have selected multiple tools
        assert len(result["selected_tool_ids"]) == len(tool_ids)
        for tool_id in tool_ids:
            assert tool_id in result["selected_tool_ids"]
        
        # Check reasoning mentions trigonometric functions
        reasoning_messages = [
            msg for msg in result["messages"] 
            if isinstance(msg, AIMessage) and "trigonometric" in msg.content
        ]
        assert len(reasoning_messages) > 0
        
    finally:
        graph_module.create_extractor = original_create_extractor


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality with improved architecture."""
    acos_tool_id = _get_acos_tool_id()
    
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor([acos_tool_id], "Async tool selection")
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(
            messages=iter([AIMessage("Async operation complete")])
        )
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Async test query")]
        })
        
        assert acos_tool_id in result["selected_tool_ids"]
        assert "messages" in result
        
        # Verify async reasoning was preserved
        reasoning_messages = [
            msg for msg in result["messages"] 
            if isinstance(msg, AIMessage) and "Async tool selection" in msg.content
        ]
        assert len(reasoning_messages) > 0
        
    finally:
        graph_module.create_extractor = original_create_extractor


def test_edge_case_empty_tool_selection():
    """Test edge case where TrustCall returns empty tool list."""
    import langsmart_bigtool.graph as graph_module
    original_create_extractor = graph_module.create_extractor
    
    def mock_create_extractor(llm, tools, tool_choice):
        return MockTrustCallExtractor([], "No tools are relevant for this query")
    
    graph_module.create_extractor = mock_create_extractor
    
    try:
        selector_llm = GenericFakeChatModel(messages=iter([]))
        main_llm = FakeMainModel(messages=iter([AIMessage("No tools scenario handled")]))
        
        builder = create_agent(selector_llm, main_llm, tool_registry)
        agent = builder.compile()
        
        result = agent.invoke({
            "messages": [HumanMessage(content="Tell me a joke")]
        })
        
        # Should handle empty selection gracefully
        assert result["selected_tool_ids"] == []
        
        # Should provide appropriate message
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert "No valid tools were selected" in final_message.content
        
    finally:
        graph_module.create_extractor = original_create_extractor