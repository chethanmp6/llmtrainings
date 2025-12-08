"""
LangChain RAG Sample Application
================================
A comprehensive RAG (Retrieval-Augmented Generation) application using
LangChain's latest version patterns.

This example demonstrates:
1. Document loading and chunking
2. Vector store indexing with embeddings
3. RAG agent with retrieval tool
4. RAG chain (two-step approach)
5. Conversation memory support

Requirements:
    pip install langchain langchain-openai langchain-community \
                langchain-text-splitters chromadb python-dotenv bs4
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """Configuration settings for the RAG application."""
    
    # Model settings
    CHAT_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K_RESULTS = 4


# ==============================================================================
# Document Loading and Processing
# ==============================================================================

def load_web_documents(urls: List[str], css_selectors: Optional[List[str]] = None):
    """
    Load documents from web URLs.
    
    Args:
        urls: List of URLs to load
        css_selectors: Optional CSS class selectors to filter content
    
    Returns:
        List of loaded documents
    """
    from langchain_community.document_loaders import WebBaseLoader
    import bs4
    
    if css_selectors:
        bs4_strainer = bs4.SoupStrainer(class_=css_selectors)
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": bs4_strainer},
        )
    else:
        loader = WebBaseLoader(web_paths=urls)
    
    return loader.load()


def load_pdf_documents(file_paths: List[str]):
    """
    Load documents from PDF files.
    
    Args:
        file_paths: List of PDF file paths
    
    Returns:
        List of loaded documents
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    return all_docs


def load_text_documents(file_paths: List[str]):
    """
    Load documents from text files.
    
    Args:
        file_paths: List of text file paths
    
    Returns:
        List of loaded documents
    """
    from langchain_community.document_loaders import TextLoader
    
    all_docs = []
    for path in file_paths:
        loader = TextLoader(path)
        all_docs.extend(loader.load())
    return all_docs


def split_documents(documents, chunk_size: int = None, chunk_overlap: int = None):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk (default from Config)
        chunk_overlap: Overlap between chunks (default from Config)
    
    Returns:
        List of document chunks
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or Config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


# ==============================================================================
# Vector Store and Embeddings
# ==============================================================================

def create_vector_store(documents, persist_directory: Optional[str] = None):
    """
    Create a vector store from documents using ChromaDB.
    
    Args:
        documents: List of document chunks
        persist_directory: Optional directory to persist the vector store
    
    Returns:
        ChromaDB vector store instance
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    
    if persist_directory:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
        )
    
    return vector_store


def load_vector_store(persist_directory: str):
    """
    Load an existing vector store from disk.
    
    Args:
        persist_directory: Directory where vector store is persisted
    
    Returns:
        ChromaDB vector store instance
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def create_in_memory_vector_store(documents):
    """
    Create an in-memory vector store (useful for testing).
    
    Args:
        documents: List of document chunks
    
    Returns:
        InMemoryVectorStore instance
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    
    embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=documents)
    return vector_store


# ==============================================================================
# RAG Chain Implementation (Simple Two-Step Approach)
# ==============================================================================

def create_rag_chain(vector_store, model_name: str = None):
    """
    Create a simple RAG chain that retrieves context and generates answers.
    
    This is a two-step approach:
    1. Retrieve relevant documents based on the query
    2. Generate answer using retrieved context
    
    Args:
        vector_store: Vector store for document retrieval
        model_name: Optional model name override
    
    Returns:
        Runnable RAG chain
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    # Initialize the LLM
    llm = ChatOpenAI(model=model_name or Config.CHAT_MODEL, temperature=0)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.TOP_K_RESULTS}
    )
    
    # Define the RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
        
Instructions:
- Use ONLY the information from the context to answer questions
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
- Be concise but comprehensive in your answers
- Cite relevant parts of the context when appropriate

Context:
{context}"""),
        ("human", "{question}"),
    ])
    
    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def create_rag_chain_with_sources(vector_store, model_name: str = None):
    """
    Create a RAG chain that returns both the answer and source documents.
    
    Args:
        vector_store: Vector store for document retrieval
        model_name: Optional model name override
    
    Returns:
        Runnable RAG chain that returns answer and sources
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(model=model_name or Config.CHAT_MODEL, temperature=0)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.TOP_K_RESULTS}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question based on the following context. If you cannot answer based on the context, say so.

Context:
{context}"""),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create chain that returns both answer and sources
    rag_chain_with_sources = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(
        answer=lambda x: (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({"context": format_docs(x["context"]), "question": x["question"]})
    )
    
    return rag_chain_with_sources


# ==============================================================================
# RAG Agent Implementation (Tool-Based Approach)
# ==============================================================================

def create_retrieval_tool(vector_store, tool_name: str = "search_knowledge_base"):
    """
    Create a retrieval tool for use with agents.
    
    Args:
        vector_store: Vector store for document retrieval
        tool_name: Name of the tool
    
    Returns:
        LangChain tool for retrieval
    """
    from langchain.tools import tool
    
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str):
        """Search the knowledge base for information relevant to the query.
        
        Use this tool when you need to find specific information to answer
        questions about the documents in the knowledge base.
        
        Args:
            query: The search query to find relevant information
        """
        retrieved_docs = vector_store.similarity_search(query, k=Config.TOP_K_RESULTS)
        serialized = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    # Rename the tool if needed
    search_knowledge_base.name = tool_name
    return search_knowledge_base


def create_rag_agent(vector_store, model_name: str = None, system_prompt: str = None):
    """
    Create a RAG agent that uses tools to retrieve information.
    
    The agent can decide when to search and can perform multiple searches
    if needed to answer complex questions.
    
    Args:
        vector_store: Vector store for document retrieval
        model_name: Optional model name override
        system_prompt: Optional custom system prompt
    
    Returns:
        LangGraph agent
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    llm = ChatOpenAI(model=model_name or Config.CHAT_MODEL, temperature=0)

    # Create the retrieval tool
    retrieval_tool = create_retrieval_tool(vector_store)
    tools = [retrieval_tool]

    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are a helpful assistant with access to a knowledge base.

Your capabilities:
- You can search the knowledge base to find relevant information
- You can perform multiple searches if needed to answer complex questions
- You should cite your sources when providing information

Guidelines:
- Always search the knowledge base when asked about specific topics
- If the first search doesn't provide enough information, try different search terms
- Be honest if you cannot find the requested information
- Provide clear, well-structured answers"""

    # Create the agent (updated for newer LangGraph version)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    
    return agent


# ==============================================================================
# Conversational RAG (With Memory)
# ==============================================================================

def create_conversational_rag_chain(vector_store, model_name: str = None):
    """
    Create a conversational RAG chain with chat history support.

    This chain can handle follow-up questions by considering
    the conversation history.

    Args:
        vector_store: Vector store for document retrieval
        model_name: Optional model name override

    Returns:
        Conversational RAG chain
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import BaseMessage

    llm = ChatOpenAI(model=model_name or Config.CHAT_MODEL, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_RESULTS})

    # Helper function to format chat history
    def format_chat_history(messages):
        if not messages:
            return ""
        formatted = []
        for message in messages:
            if hasattr(message, 'content'):
                role = "Human" if message.__class__.__name__ == "HumanMessage" else "Assistant"
                formatted.append(f"{role}: {message.content}")
        return "\n".join(formatted)

    # Helper function to contextualize questions with history
    def contextualize_question(inputs):
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        if not chat_history:
            return question

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question which might reference
context in the chat history, formulate a standalone question which can be understood
without the chat history. Do NOT answer the question, just reformulate it if needed
and otherwise return it as is.

Chat History:
{chat_history}

Latest Question: {question}"""),
        ])

        formatted_history = format_chat_history(chat_history)
        contextualized = llm.invoke(contextualize_prompt.format_messages(
            chat_history=formatted_history,
            question=question
        ))
        return contextualized.content

    # Helper function to retrieve documents
    def retrieve_docs(contextualized_question):
        return retriever.invoke(contextualized_question)

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Main RAG chain
    def rag_chain(inputs):
        contextualized_question = contextualize_question(inputs)
        docs = retrieve_docs(contextualized_question)
        context = format_docs(docs)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.

Use the following context to answer the question. If you don't know the answer based on
the context, say so.

Context:
{context}"""),
            ("human", "{input}"),
        ])

        response = llm.invoke(qa_prompt.format_messages(
            context=context,
            input=inputs["input"]
        ))

        return {"answer": response.content, "context": docs}

    return RunnableLambda(rag_chain)


class ConversationalRAG:
    """
    A conversational RAG assistant that maintains chat history.
    
    Usage:
        rag = ConversationalRAG(vector_store)
        response = rag.chat("What is machine learning?")
        response = rag.chat("Can you give me an example?")
        rag.clear_history()
    """
    
    def __init__(self, vector_store, model_name: str = None):
        """Initialize the conversational RAG assistant."""
        self.chain = create_conversational_rag_chain(vector_store, model_name)
        self.chat_history = []
    
    def chat(self, message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: User message
        
        Returns:
            Assistant response
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        response = self.chain.invoke({
            "input": message,
            "chat_history": self.chat_history,
        })
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=response["answer"]))
        
        return response["answer"]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []
    
    def get_history(self):
        """Get the current conversation history."""
        return self.chat_history


# ==============================================================================
# Utility Functions
# ==============================================================================

def query_rag(chain, question: str):
    """
    Query a RAG chain and return the response.
    
    Args:
        chain: RAG chain
        question: Question to ask
    
    Returns:
        Response string
    """
    return chain.invoke(question)


def query_agent(agent, question: str, stream: bool = False):
    """
    Query a RAG agent and return the response.
    
    Args:
        agent: RAG agent
        question: Question to ask
        stream: Whether to stream the response
    
    Returns:
        Response or generator if streaming
    """
    if stream:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            yield event["messages"][-1]
    else:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content


# ==============================================================================
# Example Usage
# ==============================================================================

def example_web_rag():
    """
    Example: RAG application for a web page (blog post).
    """
    print("=" * 60)
    print("Example: Web-based RAG")
    print("=" * 60)
    
    # 1. Load documents from web
    print("\n1. Loading documents from web...")
    docs = load_web_documents(
        urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        css_selectors=["post-title", "post-header", "post-content"]
    )
    print(f"   Loaded {len(docs)} document(s)")
    
    # 2. Split documents
    print("\n2. Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"   Created {len(chunks)} chunks")
    
    # 3. Create vector store
    print("\n3. Creating vector store...")
    vector_store = create_in_memory_vector_store(chunks)
    print("   Vector store created")
    
    # 4. Create RAG chain
    print("\n4. Creating RAG chain...")
    rag_chain = create_rag_chain(vector_store)
    print("   RAG chain ready")
    
    # 5. Query
    print("\n5. Querying...")
    question = "What is task decomposition?"
    print(f"   Question: {question}")
    answer = query_rag(rag_chain, question)
    print(f"   Answer: {answer}")
    
    return rag_chain, vector_store


def example_conversational_rag():
    """
    Example: Conversational RAG with memory.
    """
    print("\n" + "=" * 60)
    print("Example: Conversational RAG")
    print("=" * 60)
    
    # Load and process documents
    print("\n1. Setting up conversational RAG...")
    docs = load_web_documents(
        urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        css_selectors=["post-title", "post-header", "post-content"]
    )
    chunks = split_documents(docs)
    vector_store = create_in_memory_vector_store(chunks)
    
    # Create conversational RAG
    conv_rag = ConversationalRAG(vector_store)
    print("   Conversational RAG ready")
    
    # Have a conversation
    print("\n2. Starting conversation...")
    
    questions = [
        "What are the main components of an LLM-powered agent?",
        "Can you explain more about the first component?",
        "How does that relate to the other components?"
    ]
    
    for q in questions:
        print(f"\n   User: {q}")
        response = conv_rag.chat(q)
        print(f"   Assistant: {response[:500]}...")  # Truncate for display
    
    return conv_rag


def example_rag_agent():
    """
    Example: RAG agent with retrieval tool.
    """
    print("\n" + "=" * 60)
    print("Example: RAG Agent")
    print("=" * 60)
    
    # Load and process documents
    print("\n1. Setting up RAG agent...")
    docs = load_web_documents(
        urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        css_selectors=["post-title", "post-header", "post-content"]
    )
    chunks = split_documents(docs)
    vector_store = create_in_memory_vector_store(chunks)
    
    # Create agent
    agent = create_rag_agent(vector_store)
    print("   RAG agent ready")
    
    # Query agent
    print("\n2. Querying agent...")
    question = "What is Chain of Thought prompting and how does it help with task decomposition?"
    print(f"   Question: {question}")
    
    # Stream the response
    print("\n   Agent response:")
    for message in query_agent(agent, question, stream=True):
        if hasattr(message, 'content') and message.content:
            print(f"   {message.content[:800]}")  # Show content
    
    return agent


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """
    Main function demonstrating different RAG implementations.
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("LangChain RAG Sample Application")
    print("================================")
    print("This application demonstrates various RAG implementations.\n")
    
    try:
        # Run examples
        example_web_rag()
        example_conversational_rag()
        example_rag_agent()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
