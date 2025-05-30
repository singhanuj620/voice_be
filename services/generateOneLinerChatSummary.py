def generate_oneliner_summary(question: str, answer: str, llm) -> str:
    """
    Generate a one-liner summary for a question-answer pair using the LLM.
    """
    summary_prompt = f"Summarize the following Q&A in one concise line for chat history context.\nQ: {question}\nA: {answer}"
    summary_response = llm.invoke(
        [
            (
                "system",
                "You are a helpful assistant that creates concise one-line summaries for chat history.",
            ),
            ("human", summary_prompt),
        ]
    )
    return summary_response.content.strip()
