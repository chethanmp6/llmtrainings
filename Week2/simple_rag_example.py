import os
import bs4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable in .env file or export it")
        return

    print("🚀 Starting Simple RAG Example\n")

    # =========================================================================
    # Step 1: Load Documents
    # =========================================================================
    print("📄 Step 1: Loading documents from web...")
    
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs={"parse_only": bs4.SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )},
    )
    docs = loader.load()
    print(f"   Loaded {len(docs)} document(s)")

    # =========================================================================
    # Step 2: Split into Chunks
    # =========================================================================
    print("\n✂️  Step 2: Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)
    print(f"   Created {len(splits)} chunks")

    # =========================================================================
    # Step 3: Create Vector Store
    # =========================================================================
    print("\n🗃️  Step 3: Creating vector store with embeddings...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    print("   Vector store created!")

    # =========================================================================
    # Step 4: Create Retriever
    # =========================================================================
    print("\n🔍 Step 4: Setting up retriever...")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    print("   Retriever ready!")

    # =========================================================================
    # Step 5: Create RAG Chain
    # =========================================================================
    print("\n⛓️  Step 5: Building RAG chain...")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based only 
    on the following context. If you cannot answer based on the context, say so.

    Context:
    {context}"""),
        ("human", "{question}"),
    ])
    
    # Helper to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("   RAG chain ready!")

    # =========================================================================
    # Step 6: Query the RAG System
    # =========================================================================
    print("\n" + "=" * 60)
    print("💬 RAG System Ready - Let's ask some questions!")
    print("=" * 60)
    
    questions = [
        "What is Task Decomposition?",
        "What are the different approaches to task decomposition mentioned?",
        "What is Chain of Thought prompting?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 50)
        answer = rag_chain.invoke(question)
        print(f"📝 Answer: {answer}\n")
    
    # =========================================================================
    # Interactive Mode (Optional)
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎯 Interactive Mode - Ask your own questions!")
    print("   (Type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_question = input("\n❓ Your question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_question:
                continue
            
            answer = rag_chain.invoke(user_question)
            print(f"\n📝 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
