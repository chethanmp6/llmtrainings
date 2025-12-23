"""
LangGraph Checkpointer with Short Memory Example
==============================================
This example demonstrates how to use LangGraph's checkpointer functionality
for maintaining short-term memory across conversation turns with an LLM.

Features:
- Memory checkpointing for conversation persistence
- Short memory window (last N interactions)
- LLM integration with tool calling
- State management across multiple turns

Requirements:
    pip install langgraph langchain langchain-openai python-dotenv
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv(dotenv_path="../Week2/.env")

# =============================================================================
# State Definition
# =============================================================================

class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    memory_window: Annotated[int, "Number of recent interactions to keep"]
    user_preferences: Annotated[Dict[str, Any], "User preferences and context"]

# =============================================================================
# Tools Definition
# =============================================================================

@tool
def get_time() -> str:
    """Get the current time"""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def save_note(note: str, category: str = "general") -> str:
    """Save a note to memory

    Args:
        note: The note content to save
        category: Category for the note (default: general)
    """
    timestamp = datetime.now().isoformat()
    note_data = {
        "content": note,
        "category": category,
        "timestamp": timestamp
    }

    # In a real implementation, this would save to a database
    return f"Note saved: '{note}' in category '{category}' at {timestamp}"

@tool
def recall_notes(category: Optional[str] = None) -> str:
    """Recall saved notes

    Args:
        category: Optional category to filter notes
    """
    # Mock data - in real implementation, retrieve from storage
    mock_notes = [
        {"content": "Meeting at 3 PM tomorrow", "category": "schedule", "timestamp": "2024-01-15T10:00:00"},
        {"content": "Buy groceries", "category": "todo", "timestamp": "2024-01-15T11:30:00"},
        {"content": "Important project deadline next week", "category": "work", "timestamp": "2024-01-15T14:20:00"}
    ]

    filtered_notes = mock_notes
    if category:
        filtered_notes = [n for n in mock_notes if n["category"] == category]

    if not filtered_notes:
        return f"No notes found" + (f" in category '{category}'" if category else "")

    result = f"Found {len(filtered_notes)} note(s):\n"
    for note in filtered_notes:
        result += f"- [{note['category']}] {note['content']} ({note['timestamp']})\n"

    return result.strip()

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# =============================================================================
# Memory Management Functions
# =============================================================================

def trim_memory(state: ConversationState) -> ConversationState:
    """Trim conversation history to maintain short memory window"""
    messages = state["messages"]
    memory_window = state.get("memory_window", 10)  # Default to last 10 messages

    # Keep system messages and trim user/assistant interactions
    system_messages = [msg for msg in messages if msg.type == "system"]
    other_messages = [msg for msg in messages if msg.type != "system"]

    # Keep only the last N interactions (user+assistant pairs)
    if len(other_messages) > memory_window:
        other_messages = other_messages[-memory_window:]

    trimmed_messages = system_messages + other_messages

    return {
        **state,
        "messages": trimmed_messages
    }

def extract_conversation_context(state: ConversationState) -> str:
    """Extract key context from recent conversation"""
    messages = state["messages"]
    recent_messages = messages[-6:]  # Last 3 exchanges

    context_items = []
    for msg in recent_messages:
        if msg.type == "human":
            context_items.append(f"User asked: {msg.content[:100]}")
        elif msg.type == "ai" and hasattr(msg, 'content'):
            context_items.append(f"Assistant responded about: {msg.content[:100]}")

    return "Recent context: " + " | ".join(context_items) if context_items else "No recent context"

# =============================================================================
# Graph Nodes
# =============================================================================

def chat_node(state: ConversationState) -> ConversationState:
    """Main chat node that processes user input and generates responses"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize LLM with tools
    tools = [get_time, save_note, recall_notes, calculate]
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.7
    )
    llm_with_tools = llm.bind_tools(tools)

    # Get the conversation context
    context = extract_conversation_context(state)

    # Add system message with context if not present
    messages = state["messages"]
    if not any(msg.type == "system" for msg in messages):
        system_msg = SystemMessage(
            content=f"""You are a helpful assistant with short-term memory capabilities.
            You can save notes, recall information, get the current time, and perform calculations.

            {context}

            Keep responses concise and helpful. Use tools when appropriate to assist the user."""
        )
        messages = [system_msg] + messages

    # Generate response
    response = llm_with_tools.invoke(messages)

    # Add the response to messages
    updated_messages = messages + [response]

    return {
        **state,
        "messages": updated_messages
    }

def tool_node(state: ConversationState) -> ConversationState:
    """Execute tool calls and add results to messages"""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return state

    # Create tool node and execute
    tools = [get_time, save_note, recall_notes, calculate]
    tool_executor = ToolNode(tools)

    # Execute tools and get results
    tool_result = tool_executor.invoke({"messages": [last_message]})

    # Add tool results to messages
    updated_messages = messages + tool_result["messages"]

    return {
        **state,
        "messages": updated_messages
    }

def memory_cleanup_node(state: ConversationState) -> ConversationState:
    """Clean up memory by trimming old messages"""
    return trim_memory(state)

def should_use_tools(state: ConversationState) -> str:
    """Determine if tools should be used"""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "continue"

def should_continue(state: ConversationState) -> str:
    """Determine if conversation should continue"""
    last_message = state["messages"][-1]

    # If last message has tool calls, continue to get final response
    if len(state["messages"]) > 1:
        second_last = state["messages"][-2]
        if (hasattr(second_last, 'tool_calls') and second_last.tool_calls and
            last_message.type == "tool"):
            return "generate_response"

    return "end"

# =============================================================================
# Graph Construction
# =============================================================================

def create_conversation_graph() -> StateGraph:
    """Create the conversation graph with checkpointing"""

    # Initialize graph
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("memory_cleanup", memory_cleanup_node)

    # Add edges
    workflow.add_edge(START, "chat")
    workflow.add_conditional_edges(
        "chat",
        should_use_tools,
        {
            "tools": "tools",
            "continue": "memory_cleanup"
        }
    )
    workflow.add_conditional_edges(
        "tools",
        should_continue,
        {
            "generate_response": "chat",
            "end": "memory_cleanup"
        }
    )
    workflow.add_edge("memory_cleanup", END)

    return workflow

# =============================================================================
# Main Example Functions
# =============================================================================

def run_conversation_example():
    """Run a conversation example with checkpointing"""

    print("=" * 60)
    print("LangGraph Checkpointer with Short Memory Example")
    print("=" * 60)

    # Create memory saver for checkpointing
    memory = MemorySaver()

    # Compile graph with checkpointer
    workflow = create_conversation_graph()
    app = workflow.compile(checkpointer=memory)

    # Configuration for this conversation
    config = {"configurable": {"thread_id": "conversation_1"}}

    # Initial state
    initial_state = {
        "messages": [],
        "memory_window": 8,  # Keep last 8 messages
        "user_preferences": {"format": "concise"}
    }

    # Example conversation turns
    conversation_turns = [
        "Hi! Can you help me save a note that I have a meeting tomorrow at 2 PM?",
        "What time is it right now?",
        "Can you calculate 25 * 8 + 150?",
        "What notes do I have saved?",
        "Save another note about buying coffee beans, category shopping",
        "Show me all my notes again",
        "What was that calculation I asked for earlier?",
    ]

    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")

        # Add user message to state
        state = {
            **initial_state,
            "messages": [HumanMessage(content=user_input)]
        }

        # Process through graph
        result = app.invoke(state, config)

        # Get the last AI message
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if ai_messages:
            print(f"Assistant: {ai_messages[-1].content}")

        # Update initial state for next turn (simulating persistence)
        initial_state = result

        # Show memory status
        msg_count = len(result["messages"])
        print(f"(Memory: {msg_count} messages stored)")

def interactive_conversation():
    """Run an interactive conversation with checkpointing"""

    print("\n" + "=" * 60)
    print("Interactive Conversation Mode")
    print("=" * 60)
    print("Type 'quit' to exit, 'memory' to check memory status")
    print()

    # Setup
    memory = MemorySaver()
    workflow = create_conversation_graph()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "interactive_session"}}
    state = {
        "messages": [],
        "memory_window": 6,
        "user_preferences": {}
    }

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower() == 'memory':
            print(f"Memory status: {len(state['messages'])} messages")
            print(f"Memory window: {state['memory_window']}")
            continue

        if not user_input:
            continue

        # Process input
        current_state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)]
        }

        try:
            result = app.invoke(current_state, config)

            # Get assistant response
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            if ai_messages:
                print(f"Assistant: {ai_messages[-1].content}")

            # Update state
            state = result

        except Exception as e:
            print(f"Error: {str(e)}")

# =============================================================================
# Checkpointer Analysis Functions
# =============================================================================

def analyze_checkpoints():
    """Analyze saved checkpoints and memory usage"""

    print("\n" + "=" * 60)
    print("Checkpoint Analysis")
    print("=" * 60)

    memory = MemorySaver()
    workflow = create_conversation_graph()
    app = workflow.compile(checkpointer=memory)

    # Create some sample data
    config = {"configurable": {"thread_id": "analysis_thread"}}

    sample_inputs = [
        "Save a note: Project deadline next Friday",
        "What time is it?",
        "Calculate 100 / 4",
        "Show me my saved notes"
    ]

    state = {
        "messages": [],
        "memory_window": 4,
        "user_preferences": {}
    }

    print("Creating sample conversation...")
    for user_input in sample_inputs:
        current_state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)]
        }
        result = app.invoke(current_state, config)
        state = result
        print(f"  Processed: {user_input}")

    print(f"\nFinal state analysis:")
    print(f"  Total messages: {len(state['messages'])}")
    print(f"  Memory window: {state['memory_window']}")
    print(f"  Message types: {[msg.type for msg in state['messages']]}")

    # Show memory efficiency
    human_msgs = len([m for m in state['messages'] if m.type == 'human'])
    ai_msgs = len([m for m in state['messages'] if m.type == 'ai'])
    tool_msgs = len([m for m in state['messages'] if m.type == 'tool'])

    print(f"\nMessage breakdown:")
    print(f"  Human messages: {human_msgs}")
    print(f"  AI messages: {ai_msgs}")
    print(f"  Tool messages: {tool_msgs}")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        exit(1)

    try:
        # Run the conversation example
        run_conversation_example()

        # Uncomment to run interactive mode:
        # interactive_conversation()

        # Uncomment to run checkpoint analysis:
        # analyze_checkpoints()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure you have the required packages installed:")
        print("  pip install langgraph langchain langchain-openai python-dotenv")