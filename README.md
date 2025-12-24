# 🦙 PDF RAG Assistant

A local RAG (Retrieval-Augmented Generation) assistant that allows you to chat with any PDF document. Powered by **Ollama**, **LangChain**, **ChromaDB**, and **Streamlit**.

## 🚀 Features
- **100% Local**: No data leaves your machine.
- **Smart Retrieval**: Uses `nomic-embed-text` for high-quality document indexing.
- **Natural Conversations**: Powered by `llama3.2` (or your preferred Ollama model).
- **Source Transparency**: View exactly which parts of the document were used to generate the answer.

---

## 🛠️ Prerequisites

1. **Python 3.10+** (Tested on 3.13)
2. **Ollama**: [Download here](https://ollama.com/)
3. **Download Models**:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

---

## 📦 Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd pdf-rag-assistant
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   EMBED_MODEL=nomic-embed-text
   LLM_MODEL=llama3.2
   PDF_PATH=your_document.pdf
   ```

---

## 🏃 Usage

### 1. Ingest Documents
Place your PDF file in the project root and update the `PDF_PATH` in `.env` if needed. Then run:
```bash
python ingest.py
```
This will create a `chroma_db` folder containing the indexed chunks.

### 2. Run the Assistant
Start the Streamlit application:
```bash
python -m streamlit run app.py
```

---

## 📂 Project Structure
- `app.py`: The Streamlit web application.
- `ingest.py`: PDF processing and vector store creation script.
- `requirements.txt`: Python package dependencies.
- `.env`: local configuration (not tracked by git).
- `chroma_db/`: Local vector database (created after ingestion).

---

## 📜 License
MIT
