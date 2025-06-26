import math
import types
import uuid

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from langsmart_bigtool import create_agent
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

@pytest.mark.integration
def test_end_to_end_integration() -> None:
    """Test the full agent with real LLMs."""
    # Initialize LLMs
    selector_llm = init_chat_model("openai:gpt-4o-mini")
    main_llm = init_chat_model("openai:gpt-4o")

    # Create agent
    builder = create_agent(selector_llm, main_llm, tool_registry)
    agent = builder.compile()

    # Test with a real query
    query = "Use available tools to calculate the arc cosine of 0.5."
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # Assertions
    assert "messages" in result
    assert len(result["messages"]) > 1

    final_message = result["messages"][-1]
    assert "1.047" in final_message.content  # Check for the approximate result of acos(0.5)

    # Verify that a tool was selected
    assert "selected_tool_ids" in result
    assert len(result["selected_tool_ids"]) > 0
