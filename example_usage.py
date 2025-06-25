#!/usr/bin/env python3
"""
Example usage of the new LLM-driven tool selection architecture.

This example demonstrates the two-stage architecture:
1. Tool Selector: Fast LLM selects relevant tools
2. Main Agent: Uses only selected tools to execute tasks
"""

import math
import types
import uuid
from typing import Any, Dict

# Mock LLM classes for demonstration
class MockSelectorLLM:
    """Mock selector LLM that simulates tool selection."""
    
    def with_structured_output(self, schema):
        def invoke(messages):
            # Simulate intelligent tool selection based on query
            from langsmart_bigtool.graph import ToolSelectionResponse
            
            query = messages[0].content if messages else ""
            
            # Simple keyword-based selection for demo
            if "cosine" in query.lower() or "acos" in query.lower():
                return ToolSelectionResponse(
                    selected_tools=["acos_tool_id"],
                    reasoning="Selected acos tool for arc cosine calculation"
                )
            elif "sine" in query.lower():
                return ToolSelectionResponse(
                    selected_tools=["sin_tool_id"],
                    reasoning="Selected sin tool for sine calculation"
                )
            else:
                return ToolSelectionResponse(
                    selected_tools=["acos_tool_id", "sin_tool_id", "cos_tool_id"],
                    reasoning="Selected common trigonometric functions"
                )
        
        class StructuredModel:
            def invoke(self, messages):
                return invoke(messages)
            
            async def ainvoke(self, messages):
                return invoke(messages)
        
        return StructuredModel()


class MockMainLLM:
    """Mock main LLM that simulates agent execution."""
    
    def __init__(self):
        self.bound_tools = []
    
    def bind_tools(self, tools):
        self.bound_tools = tools
        return self
    
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        
        # Simulate using the bound tools
        tool_names = [tool.name for tool in self.bound_tools]
        return AIMessage(
            content=f"I have access to these tools: {', '.join(tool_names)}. "
                   f"I can help you with calculations using these functions."
        )


def create_sample_tool_registry():
    """Create a sample tool registry with math functions."""
    from langsmart_bigtool.utils import convert_positional_only_function_to_tool
    
    # Get math functions
    math_functions = [
        getattr(math, name)
        for name in dir(math)
        if isinstance(getattr(math, name), types.BuiltinFunctionType)
    ]
    
    # Convert to tools
    all_tools = []
    for function in math_functions:
        if tool := convert_positional_only_function_to_tool(function):
            all_tools.append(tool)
    
    # Create registry with predictable IDs for demo
    tool_registry = {}
    for tool in all_tools:
        if tool.name == "acos":
            tool_registry["acos_tool_id"] = tool
        elif tool.name == "sin":
            tool_registry["sin_tool_id"] = tool
        elif tool.name == "cos":
            tool_registry["cos_tool_id"] = tool
        else:
            tool_registry[str(uuid.uuid4())] = tool
    
    return tool_registry


def main():
    """Demonstrate the new architecture."""
    print("🚀 LangSmart BigTool - LLM-Driven Tool Selection Demo")
    print("=" * 60)
    
    # Create tool registry
    print("📚 Creating tool registry with math functions...")
    tool_registry = create_sample_tool_registry()
    print(f"   Created registry with {len(tool_registry)} tools")
    
    # Create LLMs
    print("\n🧠 Initializing LLMs...")
    selector_llm = MockSelectorLLM()
    main_llm = MockMainLLM()
    print("   ✓ Selector LLM (fast tool selection)")
    print("   ✓ Main LLM (task execution)")
    
    # Create agent
    print("\n🔧 Building two-stage agent...")
    from langsmart_bigtool import create_agent
    
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()
    print("   ✓ Agent compiled with two-node architecture")
    
    # Test queries
    test_queries = [
        "Calculate the arc cosine of 0.5",
        "What's the sine of π/2?",
        "Help me with trigonometry calculations"
    ]
    
    print("\n🔍 Testing tool selection with different queries...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            from langchain_core.messages import HumanMessage
            result = agent.invoke({
                "messages": [HumanMessage(content=query)]
            })
            
            print(f"   Selected tools: {result['selected_tool_ids']}")
            print(f"   Response: {result['messages'][-1].content}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed! The new architecture successfully:")
    print("   • Dynamically selects relevant tools for each query")
    print("   • Reduces context size for the main agent")
    print("   • Enables intelligent tool routing")
    print("   • Scales to large tool registries")


if __name__ == "__main__":
    main()