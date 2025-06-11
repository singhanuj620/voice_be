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
from chromadb.config import Settings
from langdetect import detect
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=512,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize ChromaDB collections (without persistence)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
settings = Settings(persist_directory="chroma_db/reports", allow_reset=True)
report_vectordb = Chroma(
    collection_name="reports",
    embedding_function=embeddings,
    persist_directory="chroma_db/reports",
    client_settings=settings,
)
chat_history_vectordb = Chroma(
    collection_name="chat_history",
    embedding_function=embeddings,
    persist_directory="chroma_db/reports",
    client_settings=settings,
)

prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt)])

# Register the tool for LLM use
chat_history_tool = ChatHistorySearchTool(chat_history_vectordb)


def is_devanagari(text):
    # Returns True if most characters are in Devanagari Unicode block
    devanagari_count = sum(
        1 for c in text if '\u0900' <= c <= '\u097F'
    )
    return devanagari_count > 0.5 * len(text) if text else False

def is_english_word(word):
    # Returns True if word is mostly ASCII letters
    return all('a' <= c.lower() <= 'z' for c in word if c.isalpha())

def is_hinglish_in_devanagari(text):
    # If text is in Devanagari but most words are English (transliterated), treat as English
    if not is_devanagari(text):
        return False
    words = text.split()
    # Simple check: if more than half the words are English (transliterated)
    english_like = sum(is_english_word(word) for word in words)
    return english_like > 0.5 * len(words) if words else False

# Hindi system prompt for LLM
hindi_system_prompt = (
    "आप एक सहायक वॉयस असिस्टेंट हैं जो उपयोगकर्ता द्वारा अपलोड की गई रिपोर्ट्स के बारे में सवालों के जवाब देता है। आपका काम रिपोर्ट्स का सारांश देना, मुख्य बिंदुओं को उजागर करना, और उपयोगकर्ता के सवालों के स्पष्ट, संक्षिप्त और संवादात्मक तरीके से उत्तर देना है।\n\nअपने उत्तरों को अच्छी तरह से संरचित और बोलचाल के लिए आसान बनाएं। अनावश्यक तकनीकी शब्दजाल या जटिल भाषा से बचें जब तक कि उपयोगकर्ता विशेष रूप से न पूछे। अपने उत्तर में पूरे प्रश्न को न दोहराएं।\n\nआप अपलोड की गई रिपोर्ट्स (PDF, Excel आदि) से डेटा की व्याख्या कर सकते हैं, प्रमुख मेट्रिक्स, ट्रेंड्स या विसंगतियों को समझा सकते हैं, और महत्वपूर्ण बिंदुओं को मानव-सुलभ तरीके से उजागर कर सकते हैं।\n\nयदि उपयोगकर्ता विशेष रूप से इनसाइट्स, ट्रेंड्स या सिफारिशें मांगता है, तो डेटा में महत्वपूर्ण ट्रेंड्स, विसंगतियों या क्रियाशील सिफारिशों को उजागर करें। अन्यथा, अपने उत्तर में इन्हें शामिल न करें।\n\nहमेशा अपने उत्तरों को ऑडियो आउटपुट के लिए अनुकूलित करें — छोटे, जानकारीपूर्ण और पेशेवर। और याद रखें, आप किसी भी प्रकार की रिपोर्ट में सहायता के लिए हैं, इसलिए उपयोगकर्ता की जरूरतों के अनुसार स्पष्टता और प्रासंगिकता बनाए रखें। जब तक विशेष रूप से अनुरोध न किया जाए, अनावश्यक विवरण या जटिलता न जोड़ें।\n\nनोट: आउटपुट संक्षिप्त और ऑडियो प्लेबैक के लिए उपयुक्त होना चाहिए। ऐसे विशेष वर्ण या फॉर्मेटिंग शामिल न करें जो स्पीच में सही से न आ पाए।"
)

def get_chat_response(
    user_input: str,
    sender,
    user_id=None,  # changed from session_id
    report_id=None,
):
    # Detect language and check for override
    def extract_language_override(text):
        # Simple regex to detect override commands
        if re.search(r"(?i)answer in english|respond in english|reply in english", text):
            return "en"
        if re.search(r"(?i)answer in hindi|respond in hindi|reply in hindi|hindi mein|hindi me|hindi mai", text):
            return "hi"
        return None

    override_lang = extract_language_override(user_input)
    try:
        detected_lang = detect(user_input)
    except Exception:
        detected_lang = "en"
    if is_hinglish_in_devanagari(user_input):
        detected_lang = "en"
    response_lang = override_lang if override_lang else detected_lang
    # Use Hindi system prompt if Hindi is requested or detected
    if response_lang == "hi":
        system_prompt_to_use = hindi_system_prompt
    else:
        system_prompt_to_use = system_prompt
    # Use search_full_report to get relevant snippet from report as context
    report_context = None
    if report_id:
        report_context = search_full_report(report_id, user_input)
    # Debug: Print all report_ids in the vector store
    all_metadatas = report_vectordb.get()["metadatas"]
    all_report_ids = set()
    for meta in all_metadatas:
        if meta and "report_id" in meta:
            all_report_ids.add(meta["report_id"])
    # Retrieve relevant docs from reports collection (chunked vector search)
    relevant_docs = report_vectordb.similarity_search(
        user_input, k=3, filter={"report_id": report_id} if report_id else None
    )
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    # Prepare messages for LLM
    messages = [("system", system_prompt_to_use)]
    if report_context:
        messages.append(("ai", f"Relevant snippet from report: {report_context}"))
    if context_text:
        messages.append(("ai", f"Context from report: {context_text}"))
    # Optionally, retrieve relevant chat history for context
    relevant_chats = chat_history_vectordb.similarity_search(user_input, k=3)
    if relevant_chats:
        chat_context = "\n".join(
            [doc.metadata.get("summary", doc.page_content) for doc in relevant_chats]
        )
        messages.append(("ai", f"Relevant chat history: {chat_context}"))
    user_input_for_llm = user_input
    # If Hindi is requested or detected, prepend explicit instruction unless already present
    if response_lang == "hi":
        # Check if user already requested answer in Hindi
        hindi_request_patterns = [
            r"उत्तर हिंदी में", r"हिंदी में जवाब", r"हिंदी में बताओ", r"हिंदी में दीजिए", r"in hindi", r"answer in hindi", r"respond in hindi", r"reply in hindi"
        ]
        if not any(re.search(pat, user_input, re.IGNORECASE) for pat in hindi_request_patterns):
            user_input_for_llm = "उत्तर हिंदी में दीजिए। " + user_input
    messages.append(("human", user_input_for_llm))
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
                "user_id": user_id,  # changed from session_id
                "report_id": report_id,
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
                "user_id": user_id,  # changed from session_id
                "report_id": report_id,
                "summary": summary,
            }
        ],
    )
    # Set TTS accent/voice based on response_lang
    if response_lang == "hi":
        accent_code = "hi-IN"
        voice_name = "hi-IN-Female"
    else:
        accent_code = "en-IN"
        voice_name = "en-IN-Female"
    mp3_bytes = synthesize_text_to_mp3(
        sanitized_response, accent_code=accent_code, voice_name=voice_name
    )
    return sanitized_response, mp3_bytes
