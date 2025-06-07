from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
import tempfile
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.full_report_store import save_full_report_text
from fastapi.responses import FileResponse
from chromadb.config import Settings

router = APIRouter()


@router.post("/upload-report-file")
def upload_report_file(
    file: UploadFile = File(...),
    userId: str = Form(None),
):
    filename = file.filename.lower()
    # Only allow PDF, Word, Excel
    if not (
        filename.endswith(".pdf")
        or filename.endswith(".docx")
        or filename.endswith(".xlsx")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, Word (.docx), and Excel (.xlsx) files are supported.",
        )
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(filename)[1]
        ) as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_path = temp_file.name
        # Load and extract text using LangChain loaders
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif filename.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        docs = loader.load()
        # Save the full text of the document using a unique report_id (e.g., filename + timestamp)
        full_text = "\n".join([doc.page_content for doc in docs])
        import time

        report_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}"
        save_full_report_text(report_id, full_text)
        # Split text into overlapping chunks for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # You can adjust chunk size
            chunk_overlap=200,  # You can adjust overlap
        )
        split_docs = text_splitter.split_documents(docs)
        # Add userId and report_id as metadata to each split_doc
        for doc in split_docs:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["userId"] = userId if userId else "user"
            doc.metadata["report_id"] = report_id
        # Store in ChromaDB using LangChain, in 'reports' collection
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_dir = "chroma_db/reports"
        collection_name = "reports"
        # Try to load existing collection, else create new
        try:
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            vectordb.add_documents(split_docs)
        except Exception:
            vectordb = Chroma.from_documents(
                split_docs,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )
        vectordb.persist()
        # Clean up temp file
        os.remove(temp_path)
        return {
            "status": "success",
            "message": "File processed and stored in vector database.",
            "report_id": report_id,
            "userId": userId if userId else "user",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-user-reports")
def get_user_reports(userId: str = Query(..., description="User ID to filter reports")):
    try:
        print(f"Fetching reports for userId: {userId}")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        settings = Settings(persist_directory="chroma_db/reports", allow_reset=True)
        vectordb = Chroma(
            collection_name="reports",
            persist_directory="chroma_db/reports",
            embedding_function=embeddings,
            client_settings=settings,
        )
        data = vectordb.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        reports = []
        matching_report_ids = set()
        for idx, id_ in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            print(f"meta.get('userId'): {meta.get('userId')}, userId param: {userId}, meta: {meta}")
            if meta.get("userId") == userId:
                matching_report_ids.add(meta.get("report_id"))
        return {"status": "success", "report_ids": list(matching_report_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-report")
def download_report(report_id: str):
    try:
        file_path = os.path.join("reports_fulltext", f"{report_id}.txt")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found.")
        return FileResponse(
            path=file_path,
            filename=f"{report_id}.txt",
            media_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
