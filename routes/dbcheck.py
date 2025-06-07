from fastapi import APIRouter, HTTPException, Query
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings
import shutil
import os

router = APIRouter()


@router.get("/dbcheck")
def db_check(userId: str = Query(None), report_id: str = Query(None)):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        settings = Settings(persist_directory="chroma_db/reports", allow_reset=True)
        vectordb = Chroma(
            collection_name="reports",
            persist_directory="chroma_db/reports",
            embedding_function=embeddings,
            client_settings=settings,
        )
        # Retrieve all documents and metadatas (ids will be included in the result)
        data = vectordb.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])  # ids should be present by default
        metadatas = data.get("metadatas", [])
        # Try to get report name from metadata if available, else use id only
        reports = []
        for idx, id_ in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            name = (
                meta.get("report_name")
                or meta.get("name")
                or meta.get("report_id")
                or id_
            )
            # Filtering logic
            if userId and meta.get("userId") != userId:
                continue
            if report_id and meta.get("report_id") != report_id:
                continue
            reports.append(
                {
                    "id": id_,
                    "name": name,
                    "userId": meta.get("userId"),
                    "report_id": meta.get("report_id"),
                }
            )
        return {"status": "success", "reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear_db")
def clear_db():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        settings = Settings(persist_directory="chroma_db/reports", allow_reset=True)
        vectordb = Chroma(
            collection_name="reports",
            persist_directory="chroma_db/reports",
            embedding_function=embeddings,
            client_settings=settings,
        )
        vectordb._client.reset()  # This clears all collections and data
    except Exception as e:
        return {"status": "error", "message": f"Chroma DB clear failed: {str(e)}"}
    # Remove reports_fulltext directory as before
    reports_path = os.path.join(os.getcwd(), "reports_fulltext")
    if os.path.exists(reports_path):
        shutil.rmtree(reports_path)
    return {
        "status": "success",
        "message": "Chroma DB cleared and reports_fulltext deleted",
    }
