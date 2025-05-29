from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from services.tts import synthesize_text_to_mp3
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import re

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
        """You are a helpful voice assistant for business reports. Your job is to summarize reports, extract key insights, and answer user questions in a clear, conversational, and concise way.

        Make sure your answers are well-structured and easy to understand when spoken aloud. Avoid excessive jargon or overly technical language unless the user asks for it. Do not repeat the entire question in your answer. Speak as if you're talking to a business professional who needs quick, insightful answers.

        You can interpret data from uploaded reports (PDFs, Excel, etc.), explain key metrics, trends, or anomalies, and highlight important points in a human-friendly way.

        Always optimize your responses for audio output — short, informative, and professional. And remember, you're here to assist with business reports, so keep the focus on clarity and relevance to the user's needs. Don't unclear answer with unnecessary details or complexity unless specifically requested.
        
        NOTE : The output should be concise and suitable for audio playback. Don't include special characters or formatting that might not translate well to speech.
        """,
    )
]


prompt_template = ChatPromptTemplate.from_messages(chat_history)


def sanitize_response(response: str) -> str:
    """
    Remove or replace special characters from the response to make it suitable for audio playback.
    """
    return re.sub(r"[\*\^\~\`\|\<\>\[\]\{\}]", "", response)


def get_chat_response(user_input: str):
    global chat_history
    print("User input:", user_input)
    print("Chat history:", chat_history)
    # Load ChromaDB and embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(embedding_function=embeddings)
    # Retrieve relevant docs from ChromaDB
    relevant_docs = vectordb.similarity_search(user_input, k=3)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    if context_text:
        # Add context as a system message before user input
        chat_history.append(AIMessage(content=f"Context from report: {context_text}"))
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    # Prepare prompt with all previous messages
    messages = chat_history.copy()
    # Generate response
    response = llm.invoke(messages)
    # Sanitize the AI response
    sanitized_response = sanitize_response(response.content)
    # Add AI response to history
    chat_history.append(AIMessage(content=sanitized_response))
    mp3_bytes = synthesize_text_to_mp3(sanitized_response)
    return sanitized_response, mp3_bytes
