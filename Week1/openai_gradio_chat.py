import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables in a file called .env

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")

# Initialize OpenAI client
# Make sure to set your API key: export OPENAI_API_KEY='your-key-here'
client = OpenAI(api_key=api_key)

def chat_with_openai(message, history):
    """
    Chat function that sends messages to OpenAI and returns responses.

    Args:
        message: The user's current message
        history: List of previous messages in the conversation

    Returns:
        The assistant's response
    """
    # Convert Gradio history format to OpenAI messages format
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})

    messages.append({"role": "user", "content": message})
    
    try:
        # Call OpenAI API
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                yield response_text
    
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure your OPENAI_API_KEY is set correctly."

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_openai,
    title="OpenAI Chat Interface",
    description="Simple chat interface powered by OpenAI's GPT models. Make sure to set your OPENAI_API_KEY environment variable.",
    examples=[
        "Hello! How are you?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming"
    ]
)

if __name__ == "__main__":
    demo.launch()
