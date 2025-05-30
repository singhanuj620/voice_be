from typing import Any
from pydantic import Field
from langchain.tools import BaseTool


class ChatHistorySearchTool(BaseTool):
    name: str = "chat_history_search"
    description: str = (
        "Searches the chat history for relevant previous messages. "
        "Input should be a query describing what to look for in the chat history."
    )
    chat_history_collection: Any = Field(exclude=True)

    def __init__(self, chat_history_collection):
        super().__init__(chat_history_collection=chat_history_collection)

    def _run(self, query: str):
        results = self.chat_history_collection.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])
