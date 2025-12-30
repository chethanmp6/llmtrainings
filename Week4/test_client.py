"""
Test Client for FastAPI LangGraph Chatbot Service
=================================================

Simple test script to interact with the chatbot API and demonstrate functionality.
"""

import requests
import json
import time
from typing import Optional

class ChatbotClient:
    """Simple client for interacting with the FastAPI chatbot service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None

    def chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """Send a message to the chatbot."""
        payload = {"content": message}
        if session_id:
            payload["session_id"] = session_id
        elif self.session_id:
            payload["session_id"] = self.session_id

        response = requests.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        self.session_id = data["session_id"]  # Store session ID for future chats
        return data

    def get_sessions(self) -> list:
        """Get all active sessions."""
        response = requests.get(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()

    def get_session_history(self, session_id: str) -> dict:
        """Get conversation history for a session."""
        response = requests.get(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> dict:
        """Delete a specific session."""
        response = requests.delete(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

    def clear_all_sessions(self) -> dict:
        """Clear all sessions."""
        response = requests.delete(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

def demo_conversation():
    """Demonstrate a sample conversation with the chatbot."""
    print("🤖 FastAPI LangGraph Chatbot Demo")
    print("=" * 50)

    client = ChatbotClient()

    # Health check
    print("🏥 Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Active Sessions: {health['active_sessions']}")
    print()

    # Sample conversation
    questions = [
        "What is RAG in machine learning?",
        "How does LangGraph work?",
        "Can you explain the concept of embeddings?",
        "What's the difference between RAG and fine-tuning?"
    ]

    print("💬 Starting Conversation:")
    print("-" * 30)

    for i, question in enumerate(questions, 1):
        print(f"👤 Question {i}: {question}")

        try:
            response = client.chat(question)
            print(f"🤖 Response: {response['response'][:200]}...")
            print(f"📅 Time: {response['timestamp']}")
            print(f"🆔 Session: {response['session_id'][:8]}...")
            print()

            # Small delay between questions
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            break

    # Show session info
    print("📊 Session Information:")
    print("-" * 25)
    try:
        sessions = client.get_sessions()
        for session in sessions:
            print(f"   Session ID: {session['session_id'][:8]}...")
            print(f"   Messages: {session['message_count']}")
            print(f"   Created: {session['created_at']}")
            print()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting sessions: {e}")

def interactive_mode():
    """Interactive chat mode."""
    print("🤖 Interactive Chat Mode")
    print("=" * 30)
    print("Type 'quit' to exit, 'history' to see conversation history")
    print()

    client = ChatbotClient()

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'history':
                if client.session_id:
                    try:
                        history = client.get_session_history(client.session_id)
                        print(f"\n📚 Conversation History (Session: {client.session_id[:8]}...):")
                        print("-" * 40)
                        for msg in history['messages']:
                            role = "👤" if msg['role'] == 'user' else "🤖"
                            print(f"{role} {msg['content']}")
                        print()
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Error getting history: {e}")
                else:
                    print("📝 No conversation history yet. Start chatting!")
                continue

            if not user_input:
                continue

            response = client.chat(user_input)
            print(f"🤖 Bot: {response['response']}")
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            print("🔧 Make sure the FastAPI server is running on http://localhost:8000")
            break

def test_api_endpoints():
    """Test all API endpoints."""
    print("🧪 Testing API Endpoints")
    print("=" * 30)

    client = ChatbotClient()

    tests = [
        ("Health Check", lambda: client.health_check()),
        ("Send Chat Message", lambda: client.chat("What is LangChain?")),
        ("Get Sessions", lambda: client.get_sessions()),
        ("Get Session History", lambda: client.get_session_history(client.session_id) if client.session_id else {"error": "No session"}),
    ]

    for test_name, test_func in tests:
        try:
            print(f"🔍 Testing {test_name}...")
            result = test_func()
            print(f"   ✅ Success: {type(result).__name__}")
            if isinstance(result, dict) and 'response' in result:
                print(f"   📝 Response length: {len(result['response'])} chars")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "demo":
            demo_conversation()
        elif mode == "interactive":
            interactive_mode()
        elif mode == "test":
            test_api_endpoints()
        else:
            print("Usage: python test_client.py [demo|interactive|test]")
    else:
        print("🤖 FastAPI LangGraph Chatbot Test Client")
        print("=" * 45)
        print("Available modes:")
        print("  demo       - Run demonstration conversation")
        print("  interactive - Interactive chat mode")
        print("  test       - Test all API endpoints")
        print()
        print("Usage: python test_client.py [mode]")
        print("Example: python test_client.py demo")