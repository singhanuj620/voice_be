from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

router = APIRouter()


@router.post("/upload-report-file")
def upload_report_file(file: UploadFile = File(...)):
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
        # Split text into overlapping chunks for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # You can adjust chunk size
            chunk_overlap=200,  # You can adjust overlap
        )
        split_docs = text_splitter.split_documents(docs)
        # Extract text from split_docs
        text_data = "\n".join([doc.page_content for doc in split_docs])
        # Store in ChromaDB using LangChain
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma.from_documents(split_docs, embedding=embeddings)
        vectordb.persist()
        # Clean up temp file
        os.remove(temp_path)
        return {
            "status": "success",
            "message": "File processed and stored in vector database.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
