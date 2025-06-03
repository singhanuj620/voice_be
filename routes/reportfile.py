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
from services.full_report_store import save_full_report_text

router = APIRouter()


@router.post("/upload-report-file")
def upload_report_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    # Allow PDF, Word, Excel, Tableau TWB, Tableau TWBX
    if not (
        filename.endswith(".pdf")
        or filename.endswith(".docx")
        or filename.endswith(".xlsx")
        or filename.endswith(".twb")
        or filename.endswith(".twbx")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, Word (.docx), Excel (.xlsx), Tableau (.twb), and Tableau (.twbx) files are supported.",
        )
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(filename)[1]
        ) as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_path = temp_file.name
        # Tableau .twbx support (packaged workbook)
        if filename.endswith(".twbx"):
            import zipfile
            import xml.etree.ElementTree as ET
            try:
                with zipfile.ZipFile(temp_path, "r") as z:
                    twb_names = [n for n in z.namelist() if n.endswith(".twb")]
                    if not twb_names:
                        os.remove(temp_path)
                        raise HTTPException(status_code=400, detail="No .twb file found inside the Tableau .twbx archive. Please upload a valid Tableau packaged workbook.")
                    twb_name = twb_names[0]
                    with z.open(twb_name) as twb_file:
                        twb_xml = twb_file.read()
                root = ET.fromstring(twb_xml)
            except Exception as e:
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail=f"Failed to parse Tableau .twbx file: {str(e)}")
            # Extract worksheet names and captions (basic)
            sheets = []
            for ws in root.findall(".//worksheet"):
                ws_name = ws.attrib.get("name", "Worksheet")
                captions = []
                for caption in ws.findall(".//caption"):
                    if caption.text:
                        captions.append(caption.text.strip())
                sheets.append(f"Worksheet: {ws_name}\n" + "\n".join(captions))
            if not sheets:
                for ws in root.findall(".//worksheet"):
                    ws_name = ws.attrib.get("name", "Worksheet")
                    sheets.append(f"Worksheet: {ws_name}")
            full_text = "\n\n".join(sheets)
            import time
            report_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}"
            save_full_report_text(report_id, full_text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            split_docs = text_splitter.create_documents([full_text])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = Chroma.from_documents(
                split_docs, embedding=embeddings, collection_name="reports"
            )
            vectordb.persist()
            os.remove(temp_path)
            return {
                "status": "success",
                "message": "Tableau .twbx file processed and stored in vector database.",
            }
        # Tableau .twb support (XML)
        if filename.endswith(".twb"):
            import xml.etree.ElementTree as ET

            try:
                with open(temp_path, "rb") as f:
                    twb_xml = f.read()
                root = ET.fromstring(twb_xml)
            except Exception as e:
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400, detail=f"Failed to parse Tableau .twb file: {str(e)}"
                )
            # Extract worksheet names and captions (basic)
            sheets = []
            for ws in root.findall(".//worksheet"):
                ws_name = ws.attrib.get("name", "Worksheet")
                captions = []
                for caption in ws.findall(".//caption"):
                    if caption.text:
                        captions.append(caption.text.strip())
                sheets.append(f"Worksheet: {ws_name}\n" + "\n".join(captions))
            if not sheets:
                for ws in root.findall(".//worksheet"):
                    ws_name = ws.attrib.get("name", "Worksheet")
                    sheets.append(f"Worksheet: {ws_name}")
            full_text = "\n\n".join(sheets)
            import time

            report_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}"
            save_full_report_text(report_id, full_text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            split_docs = text_splitter.create_documents([full_text])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = Chroma.from_documents(
                split_docs, embedding=embeddings, collection_name="reports"
            )
            vectordb.persist()
            os.remove(temp_path)
            return {
                "status": "success",
                "message": "Tableau .twb file processed and stored in vector database.",
            }
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
        # Extract text from split_docs
        text_data = "\n".join([doc.page_content for doc in split_docs])
        # Store in ChromaDB using LangChain, in 'reports' collection
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma.from_documents(split_docs, embedding=embeddings, collection_name="reports")
        vectordb.persist()
        # Clean up temp file
        os.remove(temp_path)
        return {
            "status": "success",
            "message": "File processed and stored in vector database.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
