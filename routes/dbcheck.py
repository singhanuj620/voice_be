from fastapi import APIRouter, HTTPException
from langchain_chroma import Chroma

router = APIRouter()


@router.get("/dbcheck")
def db_check():
    try:
        vectordb = Chroma(persist_directory="chroma_db")
        # Retrieve all documents, metadatas, and ids
        data = vectordb.get(include=["documents", "metadatas"])
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
