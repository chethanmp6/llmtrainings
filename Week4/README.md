# FastAPI LangGraph Chatbot Service

A production-ready FastAPI service featuring an in-memory chatbot powered by LangGraph, designed for team training and concept explanation.

## 🚀 Features

- **RESTful API**: Clean HTTP endpoints for chat functionality
- **LangGraph Integration**: State-based conversational AI workflow
- **Session Management**: In-memory conversation tracking with automatic cleanup
- **Concept Explanation**: Specialized for explaining technical concepts to teams
- **OpenAPI Documentation**: Auto-generated interactive API docs
- **CORS Support**: Ready for web frontend integration
- **Memory Management**: Configurable session limits and timeouts

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- FastAPI and dependencies (uses root requirements.txt + FastAPI)

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   # Install from the root requirements.txt
   pip install -r ../requirements.txt

   # Additional FastAPI dependencies
   pip install fastapi uvicorn[standard]
   ```

2. **Set up environment variables:**
   ```bash
   # Copy the example env file
   cp .env.example .env

   # Edit .env with your OpenAI API key
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Run the service:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## 🌐 API Endpoints

### Core Chat Endpoint

**POST `/chat`** - Send a message to the chatbot
```json
{
  "content": "What is RAG in AI?",
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "response": "RAG stands for Retrieval-Augmented Generation...",
  "session_id": "uuid-session-id",
  "timestamp": "2024-12-30T10:30:00"
}
```

### Session Management

- **GET `/sessions`** - List all active sessions
- **GET `/sessions/{session_id}`** - Get conversation history
- **DELETE `/sessions/{session_id}`** - Delete specific session
- **DELETE `/sessions`** - Clear all sessions

### Utility

- **GET `/`** - Service information and available endpoints
- **GET `/health`** - Health check and service status

## 💻 Usage Examples

### Using cURL

```bash
# Send a chat message
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"content": "Explain what LangGraph is"}'

# Get session history
curl -X GET "http://localhost:8000/sessions/your-session-id"

# Health check
curl -X GET "http://localhost:8000/health"
```

### Using Python requests

```python
import requests

# Chat with the bot
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "content": "What is the difference between RAG and fine-tuning?",
        "session_id": "my-session"
    }
)

data = response.json()
print(f"Bot: {data['response']}")
print(f"Session: {data['session_id']}")
```

### Using JavaScript/Fetch

```javascript
// Send chat message
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        content: 'How does LangChain work with vector databases?',
        session_id: 'web-session-123'
    })
});

const data = await response.json();
console.log('Response:', data.response);
```

## 🔧 Configuration

Edit the `Config` class in `main.py`:

```python
class Config:
    MODEL = "gpt-4o-mini"  # OpenAI model
    MAX_SESSIONS = 100     # Max concurrent sessions
    SESSION_TIMEOUT = 3600 # Session timeout in seconds
```

## 📚 Interactive Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

## 🧠 Chatbot Capabilities

The chatbot is optimized for explaining:

- **AI/ML Concepts**: LangChain, RAG, LangGraph, embeddings, etc.
- **Software Development**: Best practices, methodologies, frameworks
- **Technical Architecture**: System design, patterns, workflows
- **Team Training Topics**: Any technical concept your team needs to understand

### Example Conversations

**Q**: "What is RAG and why is it useful?"
**A**: "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. Think of it like having a research assistant that can quickly find relevant documents and then write a comprehensive answer based on what they found..."

**Q**: "How does LangGraph differ from LangChain?"
**A**: "Great question! While LangChain provides the building blocks for AI applications, LangGraph adds orchestration capabilities. Imagine LangChain as individual LEGO pieces, and LangGraph as the instruction manual that shows you how to build complex structures..."

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  LangGraph Agent │───▶│  OpenAI LLM     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│ In-Memory Store │    │  Session Manager │
│ (Conversations) │◄───┤  (Auto-cleanup)  │
└─────────────────┘    └──────────────────┘
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` files to version control
- **Rate Limiting**: Consider adding rate limiting for production
- **Authentication**: Add authentication for production environments
- **CORS**: Configure CORS origins for your specific frontend domains

## 🤝 Team Training Use Cases

1. **Onboarding**: New team members can ask about company tech stack
2. **Code Reviews**: Explain complex architectural decisions
3. **Learning Sessions**: Interactive Q&A during training sessions
4. **Documentation**: Get quick explanations of internal systems
5. **Best Practices**: Learn about coding standards and methodologies

## 📊 Monitoring and Debugging

- Check `/health` endpoint for service status
- Monitor active sessions via `/sessions` endpoint
- Enable LangSmith tracing for detailed conversation analysis
- Use FastAPI's built-in logging for request tracking

## 🛠️ Troubleshooting

**Common Issues:**

1. **401 Unauthorized**: Check your OpenAI API key in `.env`
2. **503 Service Unavailable**: Verify OpenAI API key has sufficient credits
3. **Memory Issues**: Reduce `MAX_SESSIONS` or `SESSION_TIMEOUT`
4. **Slow Responses**: Try using `gpt-3.5-turbo` instead of `gpt-4o-mini`

**Debug Mode:**
```bash
# Run with debug logging
LOGLEVEL=DEBUG uvicorn main:app --reload
```

## 📈 Extending the Service

### Adding New Features

1. **Persistent Storage**: Replace in-memory store with Redis/PostgreSQL
2. **Authentication**: Add JWT token authentication
3. **Webhooks**: Add webhook notifications for important conversations
4. **Analytics**: Track usage patterns and popular questions
5. **Multi-language**: Support multiple languages for global teams

### Custom System Prompts

Modify the `SYSTEM_PROMPT` variable to customize the chatbot's behavior for your specific team needs.

## 📄 License

This project is part of the Virtusa AI/ML training program. Feel free to use and modify for your team's training needs.