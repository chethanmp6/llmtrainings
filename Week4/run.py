"""
Simple runner script for the FastAPI LangGraph Chatbot Service
==============================================================

This script provides an easy way to start the chatbot service with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import langchain
        import langgraph
        from langchain_openai import ChatOpenAI
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install requirements: pip install -r ../requirements.txt")
        return False

def check_env_file():
    """Check if environment file exists and has required variables."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("📝 Please copy .env.example to .env and add your OpenAI API key")
        return False

    # Read and check for OpenAI API key
    with open(env_path) as f:
        content = f.read()
        if "OPENAI_API_KEY=" not in content or "your_openai_api_key_here" in content:
            print("❌ OpenAI API key not set in .env file")
            print("🔑 Please set OPENAI_API_KEY in .env file")
            return False

    print("✅ Environment configuration looks good")
    return True

def run_server(host="0.0.0.0", port=8000, reload=True, log_level="info"):
    """Run the FastAPI server with uvicorn."""
    cmd = [
        "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level
    ]

    if reload:
        cmd.append("--reload")

    print(f"🚀 Starting FastAPI server on http://{host}:{port}")
    print(f"📚 API documentation will be available at http://{host}:{port}/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main function to handle command line arguments and start server."""
    print("🤖 FastAPI LangGraph Chatbot Service")
    print("=" * 40)

    # Check dependencies
    if not check_requirements():
        sys.exit(1)

    # Check environment
    if not check_env_file():
        sys.exit(1)

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Usage: python run.py [options]")
            print("\nOptions:")
            print("  --host HOST      Host to bind to (default: 0.0.0.0)")
            print("  --port PORT      Port to bind to (default: 8000)")
            print("  --no-reload      Disable auto-reload")
            print("  --log-level LEVEL Log level (default: info)")
            print("\nExamples:")
            print("  python run.py                    # Start with defaults")
            print("  python run.py --port 8080        # Start on port 8080")
            print("  python run.py --no-reload        # Start without auto-reload")
            return

    # Default settings
    host = "0.0.0.0"
    port = 8000
    reload = True
    log_level = "info"

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"❌ Invalid port: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--no-reload":
            reload = False
            i += 1
        elif args[i] == "--log-level" and i + 1 < len(args):
            log_level = args[i + 1]
            i += 2
        else:
            print(f"❌ Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)

    # Start the server
    run_server(host=host, port=port, reload=reload, log_level=log_level)

if __name__ == "__main__":
    main()