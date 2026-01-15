import streamlit as st
import os
from pipeline.ingestion.pdf_loader import process_all_docs
from pipeline.processing.text_splitter import split_documents
from pipeline.embeddings.embedding_manager import EmbeddingManager
from pipeline.vectorstore.chroma_store import VectorStore
from pipeline.retrieval.retriever import RAGRetriever
from pipeline.llm.groq_llm import GroqLLM
from pipeline.agent.agentic import agentic_rag_model

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("📄 Agentic RAG Document Assistant")

UPLOAD_DIR = "data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.info("Supported file types: PDF, DOCX, DOC, XML, CSV, TXT")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf", "xml", "csv", "doc", "docx", "txt"])
file_path = None

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully")

@st.cache_resource
def load_pipeline(_):
    docs = process_all_docs(UPLOAD_DIR)
    chunks = split_documents(docs)

    embedder = EmbeddingManager()
    embeddings = embedder.generate_embeddings([c.page_content for c in chunks])

    store = VectorStore()
    store.add_documents(chunks, embeddings)

    retriever = RAGRetriever(store, embedder)
    llm = GroqLLM()
    return retriever, llm

if file_path:
    retriever, llm = load_pipeline(file_path)

    query = st.text_input("Ask a question from your documents")

    if st.button("🔍 Get Answer") and query.strip():
        answer = agentic_rag_model(query, retriever, llm)
        st.subheader("Answer")
        st.write(answer)
