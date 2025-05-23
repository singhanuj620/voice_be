from langchain_google_genai import GoogleGenerativeAIEmbeddings
from services.llm_service import llm
from langchain.schema import HumanMessage

# from langchain_community.vectorstores import Chroma
import numpy as np
import re


def generate_related_prompts(user_input: str) -> list:
    """
    Generate alternative prompts using an LLM for paraphrasing.
    """
    prompt = (
        "Paraphrase the following user prompt in 4 different ways. "
        "Return each paraphrased prompt as a list item.\n"
        f"Prompt: {user_input}"
    )
    # Use the LLM to generate paraphrased prompts
    response = llm.invoke([HumanMessage(content=prompt)])
    # Try to extract list of prompts from the response

    lines = re.split(r"\n|\r", response.content)
    # Remove empty lines and possible numbering
    paraphrased = [re.sub(r"^\d+\.\s*", "", l).strip() for l in lines if l.strip()]
    # Always include the original prompt as the first item
    if not paraphrased:
        paraphrased = [user_input]  # Fallback to original if no paraphrase generated
    else:
        paraphrased = [p for p in paraphrased if p]  # Remove empty strings
    print("Paraphrased prompts:", [user_input] + paraphrased[:4])
    return [user_input] + paraphrased[:4]


def get_best_related_prompt(user_input: str) -> str:
    # Load embeddings and ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # Generate 5 related prompts
    related_prompts = generate_related_prompts(user_input)

    # Get embedding for user_input
    user_embedding = embeddings.embed_query(user_input)

    # Get embeddings for related prompts
    related_embeddings = [embeddings.embed_query(p) for p in related_prompts]

    # Compute cosine similarity between user_input and each related prompt
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_similarity(user_embedding, emb) for emb in related_embeddings
    ]
    best_idx = int(np.argmax(similarities))
    return related_prompts[best_idx]
