from fastapi import APIRouter, HTTPException, Query
from services.llm_service import chat_history_vectordb

router = APIRouter()


@router.get("/get-user-chat-history")
def get_user_chat_history(user_id: str, report_id: str = Query(None)):
    try:
        # Build the query filter for user_id and report_id using $and if both are present
        conditions = []
        if user_id:
            conditions.append({"user_id": user_id})
        if report_id:
            conditions.append({"report_id": report_id})
        if len(conditions) == 2:
            query_filter = {"$and": conditions}
        else:
            query_filter = conditions[0] if conditions else {}
        # Query all chat history documents for this user_id and optional report_id
        results = chat_history_vectordb.get(where=query_filter)
        print(f"Query results: {results}")
        chat_history = []
        # Flatten and pair texts with their metadata
        for text, meta in zip(
            results.get("documents", []), results.get("metadatas", [])
        ):
            chat_history.append(
                {
                    "text": text,
                    "sender": meta.get("sender"),
                    "timestamp": meta.get("timestamp"),
                    "summary": meta.get("summary"),
                    "user_id": meta.get("user_id"),
                    "report_id": meta.get("report_id"),
                }
            )
        # Sort chat history by timestamp to preserve chat order
        chat_history.sort(key=lambda x: x["timestamp"])
        return {
            "user_id": user_id,
            "report_id": report_id,
            "chat_history": chat_history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
