"""
LangGraph Agent Example with Multiple Nodes and Edges
=====================================================

This example demonstrates a research assistant agent that can:
1. Process user queries
2. Search for information
3. Analyze and synthesize results
4. Provide final answers
5. Handle errors and routing decisions

The graph shows different execution paths based on user intent and processing results.
"""

import os
from typing import TypedDict, Literal, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
import json

# Load environment variables
load_dotenv()

# Configuration
class Config:
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.3
    MAX_ITERATIONS = 5

# State definition for the graph
class AgentState(TypedDict):
    messages: List[Any]
    user_query: str
    search_results: List[str]
    analysis: str
    final_answer: str
    iteration_count: int
    needs_search: bool
    error: str

# Initialize LLM
llm = ChatOpenAI(
    model=Config.MODEL,
    temperature=Config.TEMPERATURE,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Mock search tool for demonstration
@tool
def web_search(query: str) -> List[str]:
    """Search the web for information about the given query."""
    mock_results = [
        f"Search result 1 for '{query}': This is relevant information about the topic.",
        f"Search result 2 for '{query}': Additional context and details found online.",
        f"Search result 3 for '{query}': Expert opinion and analysis on the subject."
    ]
    return mock_results

# Node Functions
def query_processor(state: AgentState) -> AgentState:
    """Process and understand the user query to determine next steps."""
    print("🔍 Processing user query...")

    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        # Extract query from last human message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    # Simple intent detection
    search_keywords = ["what", "how", "why", "when", "where", "research", "find", "search"]
    needs_search = any(keyword in user_query.lower() for keyword in search_keywords)

    return {
        **state,
        "user_query": user_query,
        "needs_search": needs_search,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def search_agent(state: AgentState) -> AgentState:
    """Search for information based on the user query."""
    print("🔎 Searching for information...")

    try:
        search_results = web_search(state["user_query"])
        print(f"Found {len(search_results)} search results")

        return {
            **state,
            "search_results": search_results,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "search_results": [],
            "error": f"Search failed: {str(e)}"
        }

def analyzer_agent(state: AgentState) -> AgentState:
    """Analyze search results and synthesize information."""
    print("📊 Analyzing search results...")

    if state.get("error"):
        return state

    search_results = state.get("search_results", [])
    user_query = state.get("user_query", "")

    analysis_prompt = f"""
    Analyze the following search results for the query: "{user_query}"

    Search Results:
    {chr(10).join([f"- {result}" for result in search_results])}

    Provide a comprehensive analysis that:
    1. Identifies key themes and patterns
    2. Highlights the most relevant information
    3. Notes any gaps or conflicting information
    4. Summarizes the main insights

    Analysis:
    """

    try:
        response = llm.invoke([SystemMessage(content="You are an expert analyst."),
                              HumanMessage(content=analysis_prompt)])
        analysis = response.content

        return {
            **state,
            "analysis": analysis
        }
    except Exception as e:
        return {
            **state,
            "error": f"Analysis failed: {str(e)}"
        }

def answer_generator(state: AgentState) -> AgentState:
    """Generate the final answer based on analysis."""
    print("✍️ Generating final answer...")

    if state.get("error"):
        return {
            **state,
            "final_answer": f"I encountered an error: {state['error']}"
        }

    user_query = state.get("user_query", "")
    analysis = state.get("analysis", "")
    search_results = state.get("search_results", [])

    if not state.get("needs_search"):
        # Direct answer without search
        try:
            response = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Please answer this question directly: {user_query}")
            ])
            final_answer = response.content
        except Exception as e:
            final_answer = f"I encountered an error generating the answer: {str(e)}"
    else:
        # Answer based on search and analysis
        answer_prompt = f"""
        Based on the analysis of search results, provide a comprehensive answer to: "{user_query}"

        Analysis:
        {analysis}

        Your answer should be:
        - Clear and well-structured
        - Based on the evidence found
        - Include relevant details
        - Acknowledge any limitations or uncertainties

        Answer:
        """

        try:
            response = llm.invoke([
                SystemMessage(content="You are a knowledgeable assistant providing research-based answers."),
                HumanMessage(content=answer_prompt)
            ])
            final_answer = response.content
        except Exception as e:
            final_answer = f"I encountered an error generating the answer: {str(e)}"

    return {
        **state,
        "final_answer": final_answer
    }

# Conditional edge functions
def should_search(state: AgentState) -> Literal["search", "direct_answer"]:
    """Determine if we need to search or can answer directly."""
    if state.get("error"):
        return "direct_answer"

    return "search" if state.get("needs_search", False) else "direct_answer"

def should_continue(state: AgentState) -> Literal["analyze", "error_handler", "end"]:
    """Determine next step after search."""
    if state.get("error"):
        return "error_handler"

    search_results = state.get("search_results", [])
    if not search_results:
        return "error_handler"

    return "analyze"

def error_handler(state: AgentState) -> AgentState:
    """Handle errors and provide fallback responses."""
    print("⚠️ Handling error...")

    error_msg = state.get("error", "Unknown error occurred")
    user_query = state.get("user_query", "")

    fallback_answer = f"""
    I encountered an issue while processing your request: {error_msg}

    However, I can still try to help with your question: "{user_query}"

    Let me provide a general response based on my training knowledge...
    """

    try:
        response = llm.invoke([
            SystemMessage(content="Provide a helpful response despite technical difficulties."),
            HumanMessage(content=f"{fallback_answer}\n\nQuestion: {user_query}")
        ])
        final_answer = response.content
    except Exception:
        final_answer = "I apologize, but I'm experiencing technical difficulties and cannot process your request at this time."

    return {
        **state,
        "final_answer": final_answer
    }

# Build the graph
def create_research_agent():
    """Create and return the research agent graph."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query_processor", query_processor)
    workflow.add_node("search_agent", search_agent)
    workflow.add_node("analyzer_agent", analyzer_agent)
    workflow.add_node("answer_generator", answer_generator)
    workflow.add_node("error_handler", error_handler)

    # Add edges
    workflow.add_edge(START, "query_processor")

    # Conditional edge from query processor
    workflow.add_conditional_edges(
        "query_processor",
        should_search,
        {
            "search": "search_agent",
            "direct_answer": "answer_generator"
        }
    )

    # Conditional edge from search agent
    workflow.add_conditional_edges(
        "search_agent",
        should_continue,
        {
            "analyze": "analyzer_agent",
            "error_handler": "error_handler",
            "end": END
        }
    )

    # Linear edges
    workflow.add_edge("analyzer_agent", "answer_generator")
    workflow.add_edge("answer_generator", END)
    workflow.add_edge("error_handler", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

# Helper function to run the agent
def run_research_query(app, query: str, thread_id: str = "default") -> Dict[str, Any]:
    """Run a research query through the agent."""

    print(f"\n🚀 Starting research for: '{query}'")
    print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "search_results": [],
        "analysis": "",
        "final_answer": "",
        "iteration_count": 0,
        "needs_search": False,
        "error": ""
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run the graph
        final_state = app.invoke(initial_state, config=config)

        print("\n✅ Research completed!")
        print("=" * 60)
        print(f"Final Answer:\n{final_state['final_answer']}")

        return {
            "success": True,
            "answer": final_state["final_answer"],
            "search_results": final_state.get("search_results", []),
            "analysis": final_state.get("analysis", ""),
            "error": final_state.get("error", "")
        }

    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        print(f"\n❌ {error_msg}")
        return {
            "success": False,
            "answer": "I apologize, but I encountered an error while processing your request.",
            "error": error_msg
        }

# Demo function
def demo():
    """Demonstrate the research agent with different types of queries."""

    print("🔬 LangGraph Research Agent Demo")
    print("=" * 50)

    # Create the agent
    agent = create_research_agent()

    # Test queries
    test_queries = [
        "What is machine learning?",
        "Hello there!",
        "How does photosynthesis work in plants?",
        "What are the latest developments in quantum computing?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}")
        result = run_research_query(agent, query, thread_id=f"test_{i}")

        if result["success"]:
            print(f"✅ Success: {len(result['answer'])} characters in response")
        else:
            print(f"❌ Failed: {result['error']}")

        print("-" * 50)

if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        exit(1)

    # Run the demo
    demo()

    # Interactive mode
    print("\n🎯 Interactive Mode")
    print("Type 'quit' to exit")

    agent = create_research_agent()

    while True:
        try:
            query = input("\n💭 Enter your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if query:
                result = run_research_query(agent, query, thread_id="interactive")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")