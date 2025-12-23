import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Config
PDF_PATH = os.getenv("PDF_PATH", "your_document.pdf")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

def ingest_pdf():
    # Environment variables check
    required_vars = ["EMBED_MODEL", "OLLAMA_BASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        logger.error(f"❌ PDF file not found: {PDF_PATH}")
        return
        
    # Load PDF
    logger.info(f"📄 Loading PDF: {PDF_PATH}")
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
    except Exception as e:
        logger.error(f"❌ Failed to load PDF: {e}")
        return
    
    # Split into chunks
    logger.info("✂️ Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✅ Split into {len(chunks)} chunks")
    
    # Embeddings + Vector store
    logger.info(f"🧠 Creating embeddings using {os.getenv('EMBED_MODEL')}...")
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL")
    )
    
    # Create Chroma DB
    logger.info(f"💾 Creating vector store at {CHROMA_PATH}...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        logger.info(f"✅ Document indexed successfully into {CHROMA_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting PDF ingestion process...")
    ingest_pdf()
