import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Config
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# UI Configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

def setup_page():
    st.title("🦙 PDF RAG Assistant")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"**LLM:** `{LLM_MODEL}`")
        st.info(f"**Embedding:** `{EMBED_MODEL}`")
        st.divider()
        st.markdown("### About")
        st.markdown(
            "This assistant uses RAG (Retrieval-Augmented Generation) "
            " to answer questions based on your provided PDF document. "
            "All processing is done locally via Ollama."
        )
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False)
def create_rag_chain(_retriever):
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.2,
            num_ctx=4096
        )
        
        system_prompt = (
            "You are a helpful assistant answering questions about the SML script PDF. "
            "Use only the provided context to answer. If the answer isn't in the context, "
            "say 'I don't have enough information to answer that question based on the document.'"
            "\n\nContext:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        doc_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(_retriever, doc_chain)
    except Exception as e:
        st.error(f"❌ Error creating RAG chain: {str(e)}")
        st.stop()

def main():
    setup_page()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Load models
    with st.spinner("🔍 Initializing models..."):
        retriever = load_retriever()
        rag_chain = create_rag_chain(retriever)
    
    # Chat input
    if user_input := st.chat_input("Ask a question about your document..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    sources = response["context"]
                    
                    st.markdown(answer)
                    
                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(sources):
                            page = doc.metadata.get('page', 'N/A')
                            st.markdown(f"**Source {i+1}** (Page {page}):")
                            st.caption(doc.page_content[:400] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
