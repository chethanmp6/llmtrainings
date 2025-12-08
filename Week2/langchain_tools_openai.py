"""
LangChain Tools Example (OpenAI)
================================
This example demonstrates how to create and use tools with LangChain,
allowing an LLM to interact with external functions and APIs.

Requirements:
    pip install langchain langchain-openai python-dotenv
"""

import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Load environment variables
load_dotenv()


# =============================================================================
# Define Custom Tools
# =============================================================================

@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location.
    
    Args:
        location: The city and country, e.g., "London, UK"
        unit: Temperature unit - "celsius" or "fahrenheit"
    """
    # Simulated weather data (in real use, call a weather API)
    weather_data = {
        "london, uk": {"temp": 15, "condition": "Cloudy", "humidity": 75},
        "new york, usa": {"temp": 22, "condition": "Sunny", "humidity": 45},
        "tokyo, japan": {"temp": 28, "condition": "Humid", "humidity": 80},
    }
    
    location_key = location.lower()
    data = weather_data.get(location_key, {"temp": 20, "condition": "Unknown", "humidity": 50})
    
    temp = data["temp"]
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
        unit_symbol = "°F"
    else:
        unit_symbol = "°C"
    
    return f"Weather in {location}: {temp}{unit_symbol}, {data['condition']}, Humidity: {data['humidity']}%"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate, e.g., "2 + 2 * 3"
    """
    try:
        # Using eval with restricted globals for safety
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """Get the current date and time.
    
    Args:
        timezone: Optional timezone name (currently returns UTC)
    """
    now = datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_database(query: str, limit: int = 5) -> str:
    """Search a mock database for information.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
    """
    # Simulated database search
    mock_data = [
        {"id": 1, "title": "Introduction to Python", "category": "Programming"},
        {"id": 2, "title": "Machine Learning Basics", "category": "AI"},
        {"id": 3, "title": "Web Development Guide", "category": "Programming"},
        {"id": 4, "title": "Data Science Handbook", "category": "Data"},
        {"id": 5, "title": "Neural Networks Deep Dive", "category": "AI"},
    ]
    
    query_lower = query.lower()
    results = [
        item for item in mock_data 
        if query_lower in item["title"].lower() or query_lower in item["category"].lower()
    ][:limit]
    
    if not results:
        return f"No results found for '{query}'"
    
    return f"Found {len(results)} results:\n" + "\n".join(
        f"  - {r['title']} ({r['category']})" for r in results
    )


# =============================================================================
# Main Example
# =============================================================================

def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("Set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("=" * 60)
        return
    
    # Initialize the model with tools
    tools = [get_current_weather, calculate, get_current_time, search_database]
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
    )
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)
    
    print("=" * 60)
    print("LangChain Tools Demo (OpenAI)")
    print("=" * 60)
    print("\nAvailable tools:")
    for t in tools:
        print(f"  • {t.name}: {t.description.split('.')[0]}")
    print()
    
    # Example queries that will trigger tool usage
    queries = [
        "What's the weather like in Tokyo, Japan?",
        "Calculate 15 * 7 + 23",
        "What time is it right now?",
        "Search for AI-related content in the database",
    ]
    
    # Tool lookup dictionary
    tool_map = {t.name: t for t in tools}
    
    for query in queries:
        print(f"User: {query}")
        print("-" * 40)
        
        # Get response with potential tool calls
        response = llm_with_tools.invoke([HumanMessage(content=query)])
        
        # Check if the model wants to use tools
        if response.tool_calls:
            print(f"Tool calls requested: {len(response.tool_calls)}")
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"  → Calling: {tool_name}({tool_args})")
                
                # Execute the tool
                tool_fn = tool_map.get(tool_name)
                if tool_fn:
                    result = tool_fn.invoke(tool_args)
                    print(f"  ← Result: {result}")
        else:
            print(f"Response: {response.content}")
        
        print()


# =============================================================================
# Full Conversation Loop with Tool Execution
# =============================================================================

def conversation_with_tools():
    """Complete example showing the full tool-calling loop."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example")
        return
    
    tools = [get_current_weather, calculate, get_current_time, search_database]
    tool_map = {t.name: t for t in tools}
    
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)
    
    print("=" * 60)
    print("Full Conversation Loop Example")
    print("=" * 60)
    
    # Start a conversation
    query = "What's the weather in New York and London? Also calculate 100/4 + 50"
    messages = [HumanMessage(content=query)]
    
    print(f"\nUser: {query}\n")
    
    # First call - model decides what tools to use
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    # Process tool calls if any
    while response.tool_calls:
        print(f"Model requested {len(response.tool_calls)} tool call(s):")
        
        # Execute each tool and add results to messages
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"  → {tool_name}({tool_args})")
            
            # Execute tool
            result = tool_map[tool_name].invoke(tool_args)
            print(f"  ← {result}")
            
            # Add tool result to messages
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))
        
        # Get next response (might have more tool calls or final answer)
        response = llm_with_tools.invoke(messages)
        messages.append(response)
    
    # Final response
    print(f"\nAssistant: {response.content}")


# =============================================================================
# Agent Example (Autonomous Operation)
# =============================================================================

def agent_example():
    """Demonstrates using tools with a LangChain agent for autonomous operation."""
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example")
        return
    
    # Setup
    tools = [get_current_weather, calculate, get_current_time, search_database]
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    
    # Create a prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools. "
                   "Use them when needed to answer questions accurately."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("=" * 60)
    print("Agent Example")
    print("=" * 60)
    
    # The agent will automatically decide which tools to use
    result = agent_executor.invoke({
        "input": "I'm planning a trip. What's the weather in London and "
                 "what time is it there? Also, find me some AI learning resources."
    })
    
    print("\nFinal Answer:", result["output"])


if __name__ == "__main__":
    main()
    
    # Uncomment to run additional examples:
    # conversation_with_tools()
    # agent_example()
