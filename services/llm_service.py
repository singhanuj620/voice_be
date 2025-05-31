from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from services.tts import synthesize_text_to_mp3
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime
import os
from services.generateOneLinerChatSummary import generate_oneliner_summary
from services.chatHistoryTool import ChatHistorySearchTool
from services.systemPrompt import system_prompt
from services.sanatizeResponse import sanitize_response
from services.full_report_search import search_full_report

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=512,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize ChromaDB collections (without persistence)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
report_vectordb = Chroma(collection_name="reports", embedding_function=embeddings)
chat_history_vectordb = Chroma(
    collection_name="chat_history", embedding_function=embeddings
)

prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt)])

# Register the tool for LLM use
chat_history_tool = ChatHistorySearchTool(chat_history_vectordb)


def get_chat_response(user_input: str, sender="user", session_id=None, report_id=None, accent_code="en-IN", voice_name="en-IN-Wavenet-A"):
    print("User input:", user_input)
    # Try global search in full report text first if report_id is provided
    global_context = None
    if report_id:
        global_context = search_full_report(report_id, user_input)
    # Retrieve relevant docs from reports collection (chunked vector search)
    relevant_docs = report_vectordb.similarity_search(user_input, k=3)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    # Prepare messages for LLM
    messages = [("system", system_prompt)]
    if global_context:
        messages.append(("ai", f"Global context from report: {global_context}"))
    if context_text:
        messages.append(("ai", f"Context from report: {context_text}"))
    # Optionally, retrieve relevant chat history for context
    relevant_chats = chat_history_vectordb.similarity_search(user_input, k=3)
    if relevant_chats:
        chat_context = "\n".join(
            [doc.metadata.get("summary", doc.page_content) for doc in relevant_chats]
        )
        messages.append(("ai", f"Relevant chat history: {chat_context}"))
    print("Relevant chat history:", relevant_chats)
    messages.append(("human", user_input))
    print("Messages for LLM:", messages)
    # Generate response
    response = llm.invoke(messages, tools=[chat_history_tool])
    print("##LLM response:", response.content)
    sanitized_response = sanitize_response(response.content)
    # Generate one-liner summary for this Q&A
    summary = generate_oneliner_summary(user_input, sanitized_response, llm)
    # Store user message in chat_history collection (with summary=None)
    chat_history_vectordb.add_texts(
        texts=[user_input],
        metadatas=[
            {
                "sender": sender,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "summary": None,
            }
        ],
    )
    # Store AI response in chat_history collection (with summary)
    chat_history_vectordb.add_texts(
        texts=[sanitized_response],
        metadatas=[
            {
                "sender": "ai",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "summary": summary,
            }
        ],
    )
    print("@@LLM response:", sanitized_response)
    mp3_bytes = synthesize_text_to_mp3(sanitized_response, accent_code=accent_code, voice_name=voice_name)
    return sanitized_response, mp3_bytes
