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
from tableauhyperapi import TableName

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
            data_text = ""
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
                    # --- Extract .hyper files and read data ---
                    hyper_names = [n for n in z.namelist() if n.endswith(".hyper")]
                    if hyper_names:
                        try:
                            from tableauhyperapi import HyperProcess, Connection, Telemetry, TableName
                            import tempfile as _tempfile
                            for hyper_name in hyper_names:
                                with z.open(hyper_name) as hyper_file:
                                    with _tempfile.NamedTemporaryFile(delete=False, suffix=".hyper") as hyper_temp:
                                        hyper_temp.write(hyper_file.read())
                                        hyper_temp.flush()
                                        hyper_path = hyper_temp.name
                                with HyperProcess(telemetry=Telemetry.SEND_USAGE_DATA_TO_TABLEAU) as hyper:
                                    with Connection(endpoint=hyper.endpoint, database=hyper_path) as connection:
                                        schema_names = connection.catalog.get_schema_names()
                                        for schema in schema_names:
                                            tables = connection.catalog.get_table_names(schema)
                                            for table in tables:
                                                try:
                                                    # Log schema and table for debugging
                                                    print(f"[DEBUG] Extracting table: schema='{schema}', table='{table}'")
                                                    # Use TableName(table) if schema is empty or schema == table
                                                    if not schema or schema == table:
                                                        table_name = TableName(table)
                                                    else:
                                                        table_name = TableName(schema, table)
                                                    rows = connection.execute_list_query(f'SELECT * FROM {table_name} LIMIT 20')
                                                    data_text += f"Table: {schema}.{table}\n"
                                                    for row in rows:
                                                        data_text += str(row) + "\n"
                                                    data_text += "\n"
                                                except Exception as e:
                                                    data_text += f"[Could not extract data from table {table}: {str(e)}]\n"
                                os.remove(hyper_path)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            data_text += f"[Could not extract .hyper data: {str(e)}\n{tb}]"
                    # --- End .hyper extraction ---
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
            if data_text:
                full_text += "\n\nExtracted Data (first 20 rows per table):\n" + data_text
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
                "message": "Tableau .twbx file processed and stored in vector database (with extracted data if available).",
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
            docs = loader.load()
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
            docs = loader.load()
        elif filename.endswith(".xlsx"):
            try:
                print(f"[DEBUG] Attempting to load .xlsx file: {temp_path}")
                loader = UnstructuredExcelLoader(temp_path)
                docs = loader.load()
                print(f"[DEBUG] Loaded {len(docs)} document(s) from .xlsx file.")
                if not docs or all(not doc.page_content.strip() for doc in docs):
                    print(f"[DEBUG] No content extracted from .xlsx file: {temp_path}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ERROR] Failed to load .xlsx file: {e}\n{tb}")
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail=f"Failed to parse .xlsx file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        # Save the full text of the document using a unique report_id (e.g., filename + timestamp)
        full_text = "\n".join([doc.page_content for doc in docs])
        print(f"[DEBUG] Full text extracted from file: {full_text[:500]}...")
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
