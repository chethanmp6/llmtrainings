# LangChain RAG Sample Application

A comprehensive Retrieval-Augmented Generation (RAG) application using LangChain's latest version (0.3.x).

## Overview

This project demonstrates various RAG implementations using LangChain:

1. **Simple RAG Chain** - Basic two-step approach (retrieve → generate)
2. **RAG with Sources** - Returns both answer and source documents
3. **RAG Agent** - Tool-based approach that can perform multiple searches
4. **Conversational RAG** - Maintains chat history for follow-up questions

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY='your-openai-api-key'

# Optional: Enable LangSmith tracing
export LANGSMITH_TRACING='true'
export LANGSMITH_API_KEY='your-langsmith-api-key'
```

### 3. Run the Simple Example

```bash
python simple_rag_example.py
```

### 4. Run the Full Application

```bash
python langchain_rag_sample.py
```

## Project Structure

```
├── langchain_rag_sample.py   # Full-featured RAG application
├── simple_rag_example.py     # Minimal standalone example
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Components

### Document Loading

```python
from langchain_community.document_loaders import WebBaseLoader

# Load from web
docs = load_web_documents(
    urls=["https://example.com/article"],
    css_selectors=["article-content"]
)

# Load from PDF
docs = load_pdf_documents(["document.pdf"])

# Load from text files
docs = load_text_documents(["file.txt"])
```

### Document Splitting

```python
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)
```

### Vector Store

```python
# In-memory (for testing)
vector_store = create_in_memory_vector_store(chunks)

# Persistent (ChromaDB)
vector_store = create_vector_store(chunks, persist_directory="./chroma_db")

# Load existing
vector_store = load_vector_store("./chroma_db")
```

### RAG Chain (Simple)

```python
rag_chain = create_rag_chain(vector_store)
answer = rag_chain.invoke("What is machine learning?")
```

### RAG Chain with Sources

```python
rag_chain = create_rag_chain_with_sources(vector_store)
result = rag_chain.invoke("What is machine learning?")
print(result["answer"])
print(result["context"])  # Source documents
```

### RAG Agent (Tool-based)

```python
agent = create_rag_agent(vector_store)

# Query with streaming
for message in query_agent(agent, "Explain task decomposition", stream=True):
    print(message.content)
```

### Conversational RAG

```python
conv_rag = ConversationalRAG(vector_store)

# First question
response1 = conv_rag.chat("What is machine learning?")

# Follow-up (uses context from previous question)
response2 = conv_rag.chat("Can you give me an example?")

# Clear history for new conversation
conv_rag.clear_history()
```

## Configuration

Modify the `Config` class to customize settings:

```python
class Config:
    CHAT_MODEL = "gpt-4o-mini"           # or "gpt-4o", "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 4
```

## Using Different LLM Providers

### OpenAI (Default)

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

### Anthropic Claude

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
```

### Local Models (Ollama)

```bash
pip install langchain-ollama
```

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
```

## Using Different Vector Stores

### FAISS

```bash
pip install faiss-cpu
```

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

# Load later
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

### Pinecone

```bash
pip install pinecone-client langchain-pinecone
```

```python
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents, 
    embeddings,
    index_name="my-index"
)
```

## Advanced Usage

### Custom Retrieval Tool

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def custom_search(query: str, category: str = None):
    """Search with optional category filter."""
    filter_dict = {"category": category} if category else None
    docs = vector_store.similarity_search(query, k=4, filter=filter_dict)
    return format_docs(docs), docs
```

### Hybrid Search (Semantic + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = vector_store.as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

### Re-ranking Results

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank(top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
2. **Import Error**: Run `pip install -r requirements.txt`
3. **Memory Error**: Reduce `CHUNK_SIZE` or use fewer documents
4. **Slow Response**: Try a smaller model like `gpt-4o-mini`

### Debug Mode

Enable LangSmith tracing to debug chains:

```bash
export LANGSMITH_TRACING='true'
export LANGSMITH_API_KEY='your-key'
```

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/) - For tracing and debugging

## License

MIT License - Feel free to use and modify for your projects.
