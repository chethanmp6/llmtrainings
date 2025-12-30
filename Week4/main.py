"""
FastAPI LangGraph Chatbot Service
=================================

An in-memory chatbot service using FastAPI and LangGraph for team training demonstrations.
This service provides concept explanation capabilities with conversation memory.

Features:
- RESTful API endpoints
- In-memory conversation storage
- LangGraph-powered conversational AI
- Session management
- Concept explanation focused responses

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configuration
class Config:
    MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAX_SESSIONS = 100  # Limit concurrent sessions
    SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Chatbot Service",
    description="In-memory chatbot for concept explanation and team training",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    content: str = Field(..., description="The message content")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int

class ConversationState(BaseModel):
    messages: List[Dict] = Field(default_factory=list)
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = Field(default=0)

# In-memory storage
conversations: Dict[str, ConversationState] = {}

# LangGraph State Definition
class ChatbotState(BaseModel):
    messages: List = Field(default_factory=list)
    user_input: str = ""
    response: str = ""

# Initialize LLM
llm = ChatOpenAI(
    model=Config.MODEL,
    api_key=Config.OPENAI_API_KEY,
    temperature=0.7
)

# System prompt for concept explanation
SYSTEM_PROMPT = """You are an AI assistant designed to explain technical concepts to office team members in a clear, engaging way.

Your role is to:
1. Break down complex topics into digestible explanations
2. Use analogies and real-world examples when helpful
3. Adapt your explanation level based on the context
4. Be encouraging and supportive of learning
5. Ask clarifying questions when needed
6. Provide practical examples and use cases

Focus areas include:
- AI/ML concepts (LangChain, RAG, LangGraph, etc.)
- Software development practices
- Technology explanations
- Best practices and methodologies

Keep responses concise but thorough, and always maintain a friendly, professional tone.
"""

def create_chatbot_graph():
    """Create and configure the LangGraph chatbot."""

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | llm

    def process_message(state: dict):
        """Process user message and generate response."""
        try:
            # Get the last user message
            user_input = state.get("user_input", "")

            # Generate response
            response = chain.invoke({"input": user_input})

            # Update state
            state["response"] = response.content
            return state

        except Exception as e:
            state["response"] = f"I apologize, but I encountered an error: {str(e)}"
            return state

    # Create the graph
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("process", process_message)

    # Add edges
    workflow.add_edge(START, "process")
    workflow.add_edge("process", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Initialize the chatbot graph
chatbot_graph = create_chatbot_graph()

# Utility functions
def cleanup_old_sessions():
    """Remove old inactive sessions."""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session in conversations.items()
        if (current_time - session.last_activity).seconds > Config.SESSION_TIMEOUT
    ]

    for session_id in expired_sessions:
        del conversations[session_id]

def get_or_create_session(session_id: Optional[str]) -> str:
    """Get existing session or create a new one."""
    if session_id and session_id in conversations:
        # Update last activity
        conversations[session_id].last_activity = datetime.now()
        return session_id

    # Create new session
    new_session_id = str(uuid.uuid4())
    conversations[new_session_id] = ConversationState(session_id=new_session_id)

    # Cleanup if too many sessions
    if len(conversations) > Config.MAX_SESSIONS:
        cleanup_old_sessions()

    return new_session_id

# API Endpoints
@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "message": "LangGraph Chatbot Service is running!",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "sessions": "/sessions",
            "health": "/health"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for conversing with the chatbot.

    - **content**: The message to send to the chatbot
    - **session_id**: Optional session ID for conversation continuity
    """
    try:
        # Get or create session
        session_id = get_or_create_session(message.session_id)
        session = conversations[session_id]

        # Prepare the graph state
        config = {"configurable": {"thread_id": session_id}}

        # Invoke the chatbot
        result = chatbot_graph.invoke(
            {"user_input": message.content},
            config=config
        )

        # Update session
        session.message_count += 1
        session.last_activity = datetime.now()

        # Store messages in session
        session.messages.extend([
            {"role": "user", "content": message.content, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": result["response"], "timestamp": datetime.now().isoformat()}
        ])

        # Schedule cleanup
        background_tasks.add_task(cleanup_old_sessions)

        return ChatResponse(
            response=result["response"],
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """Get information about all active sessions."""
    return [
        SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            last_activity=session.last_activity,
            message_count=session.message_count
        )
        for session in conversations.values()
    ]

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Get conversation history for a specific session."""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")

    session = conversations[session_id]
    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "last_activity": session.last_activity,
        "message_count": session.message_count,
        "messages": session.messages
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its history."""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")

    del conversations[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all sessions and conversation history."""
    conversations.clear()
    return {"message": "All sessions cleared successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(conversations),
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)