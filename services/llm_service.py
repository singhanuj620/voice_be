from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from services.tts import synthesize_text_to_mp3
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=512,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Function to handle dynamic chat with history
chat_history = [
    (
        "system",
        """You are a helpful cricket assistant. which can answer questions about cricket. You can also provide information about cricket players, teams, and matches. Also you can provide information about cricket rules and regulations. You can also provide information about cricket history and statistics.
        Make sure the output is in such a way that it can be converted to audio.
        """,
    ),
]

prompt_template = ChatPromptTemplate.from_messages(chat_history)


def get_chat_response(user_input: str):
    global chat_history
    print("User input:", user_input)
    print("Chat history:", chat_history)
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    # Prepare prompt with all previous messages
    messages = chat_history.copy()
    # Generate response
    response = llm.invoke(messages)
    # Add AI response to history
    chat_history.append(AIMessage(content=response.content))
    mp3_bytes = synthesize_text_to_mp3(response.content)
    return response.content, mp3_bytes
