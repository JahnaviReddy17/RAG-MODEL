# pyrefly: ignore [missing-import]
import streamlit as st
import os
import shutil
import hashlib
from pipeline.ingestion.pdf_loader import process_all_docs
from pipeline.processing.text_splitter import split_documents
from pipeline.embeddings.embedding_manager import EmbeddingManager
from pipeline.vectorstore.chroma_store import VectorStore
from pipeline.retrieval.retriever import RAGRetriever
from pipeline.llm.groq_llm import GroqLLM
from pipeline.agent.agentic import agentic_rag_model

# 1. Page Configuration & Aesthetic Injection
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Style Sheet Injection
st.markdown("""
<style>
    /* Elegant gradient heading */
    .title-container {
        text-align: center;
        padding: 1.5rem 0rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #71717a;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    /* Sleek card container */
    .status-card {
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(228, 228, 231, 0.8);
        background-color: rgba(255, 255, 255, 0.6);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    /* Dark mode override for card container */
    @media (prefers-color-scheme: dark) {
        .status-card {
            border: 1px solid rgba(63, 63, 70, 0.7);
            background-color: rgba(24, 24, 27, 0.6);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
    }
    /* Text highlighted badges */
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #6366f1;
        color: white;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #10b981;
        color: white;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Directory Configurations
UPLOAD_DIR = "data/uploaded_pdfs"
DEMO_DIR = "data/demo_files"
VECTOR_DIR = "data/vector_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# 3. Configurations & Inputs (Main Panel)
groq_api_env = os.getenv("GROQ_API_KEY")
api_key = groq_api_env

# Default RAG settings
top_k = 4
min_score = 0.15
model_name = "llama-3.1-8b-instant"

# Header Logo & Title
st.markdown("""
<div class="title-container">
    <h1 class="main-title">🤖 Agentic RAG Document Assistant</h1>
    <p class="subtitle">Structured Knowledge Retrieval & Strict Source-Grounded Fact Extraction</p>
</div>
""", unsafe_allow_html=True)

# 3.1 API Key checking in main dashboard if not loaded from .env
if not api_key:
    st.warning("⚠️ Groq API Key not detected in `.env` environment file.")
    api_key = st.text_input("Enter your Groq API Key to proceed:", type="password", help="Get a free key from console.groq.com")
    if api_key:
        st.success("🔑 API Key configured successfully from input!")

# 3.2 Dataset source selection (horizontal at the top of the main panel)
dataset_source = st.radio(
    "📂 Choose Document Corpus Source:",
    ("💡 Use Preloaded Demo", "📁 Upload Custom Files"),
    horizontal=True
)

# Handle copying of the preloaded demo if selected
if dataset_source == "💡 Use Preloaded Demo":
    demo_file = os.path.join(DEMO_DIR, "agentic_rag_guide.txt")
    target_demo_file = os.path.join(UPLOAD_DIR, "agentic_rag_guide.txt")
    
    # Check if demo file exists, if not write a default copy
    if not os.path.exists(demo_file):
        os.makedirs(DEMO_DIR, exist_ok=True)
        # Fallback default if not already created
        with open(demo_file, "w", encoding="utf-8") as f:
            f.write("# Fallback Agentic RAG Guide\nThis is a sample document for testing the RAG pipeline.")
    
    # Synchronize the upload dir to ONLY contain the demo file to avoid overlapping context
    if not os.path.exists(target_demo_file) or len(os.listdir(UPLOAD_DIR)) > 1:
        # Clear existing uploads
        for f_name in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, f_name))
            except Exception:
                pass
        shutil.copy(demo_file, target_demo_file)
        # Invalidate the cache to force reloading
        st.cache_resource.clear()
else:
    # If custom documents selected, remove the preloaded demo guide from the uploads folder to keep database clean
    demo_guide_path = os.path.join(UPLOAD_DIR, "agentic_rag_guide.txt")
    if os.path.exists(demo_guide_path):
        try:
            os.remove(demo_guide_path)
            st.cache_resource.clear()
            st.rerun()
        except Exception:
            pass

# 4. Helper to compute cache keys based on UPLOAD_DIR state
def get_directory_state_hash(dir_path):
    files = sorted(os.listdir(dir_path))
    state_str = ""
    for f in files:
        full_path = os.path.join(dir_path, f)
        if os.path.isfile(full_path):
            state_str += f"{f}_{os.path.getmtime(full_path)}_{os.path.getsize(full_path)}|"
    return hashlib.md5(state_str.encode()).hexdigest()

# 5. Core Pipeline Ingestion & Vector DB setup (Cached for ultra-performance)
@st.cache_resource
def build_rag_system(directory_hash, _api_key, _model_name):
    # Ensure there are files to load
    files = os.listdir(UPLOAD_DIR)
    if not files:
        return None
    
    with st.spinner("⚡ Initializing RAG Pipeline (Parsing files, chunking, generating embeddings & vector database)..."):
        # A. Ingest all docs
        docs = process_all_docs(UPLOAD_DIR)
        if not docs:
            return None
        
        # B. Split documents
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        
        # C. Embedding model instantiation
        embedder = EmbeddingManager()
        embeddings = embedder.generate_embeddings([c.page_content for c in chunks])
        
        # D. Add to vector store (Fresh VectorStore in each cache run)
        store = VectorStore(persist_dir=VECTOR_DIR)
        store.clear_collection()
        store.add_documents(chunks, embeddings)
        
        # E. Setup retriever & LLM
        retriever = RAGRetriever(store, embedder)
        llm = GroqLLM(model=_model_name, api_key=_api_key)
        
        return {
            "retriever": retriever,
            "llm": llm,
            "chunks_count": len(chunks),
            "docs_count": len(docs)
        }

# Get directory state for dynamic caching
dir_state_hash = get_directory_state_hash(UPLOAD_DIR)

# Attempt pipeline initialization
rag_system = None
if api_key:
    try:
        rag_system = build_rag_system(dir_state_hash, api_key, model_name)
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")

# 6. Dashboard Interface Layout
if not api_key:
    st.info("🔑 Please enter a valid **Groq API Key** above to activate the Agentic LLM pipeline.")

else:
    # Set up interactive tabs (removed System Architecture tab)
    tab_chat, tab_corpus, tab_inspector = st.tabs([
        "💬 RAG Chat Assistant", 
        "📁 Document Corpus", 
        "🔍 Retrieval Inspector"
    ])
    
    # ------------------ TAB 1: RAG CHAT ASSISTANT ------------------
    with tab_chat:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 💡 Quick Sample Questions")
            st.markdown("Click any of the questions below to quickly test the pipeline against the document.")
            
            sample_questions = []
            if dataset_source == "💡 Use Preloaded Demo":
                sample_questions = [
                    "What is Agentic RAG and how does it solve passive RAG limits?",
                    "Explain the core architectural modules of this project.",
                    "What are the optimal chunking settings for RAG?",
                    "What is self-correction or Self-RAG?"
                ]
            else:
                sample_questions = [
                    "What is the main subject or purpose of this document?",
                    "Who is mentioned in this document and what are their details?",
                    "Summarize the key information or marks/grades listed here."
                ]
            
            selected_query = ""
            for i, sq in enumerate(sample_questions):
                if st.button(f"❓ {sq}", key=f"sq_{i}", use_container_width=True):
                    selected_query = sq
        
        with col1:
            st.markdown("### 💬 Conversational RAG Query")
            
            # Text area for question input
            user_input = st.text_input(
                "Type your query below:",
                value=selected_query,
                placeholder="Ask something about the ingested documents..."
            )
            
            # Check pipeline readiness
            if not rag_system:
                if len(os.listdir(UPLOAD_DIR)) == 0:
                    st.warning("⚠️ No documents loaded! Please upload a file in the 'Document Corpus' tab or choose the Preloaded Demo.")
                else:
                    st.warning("⚠️ Ready to load system. Please interact or trigger.")
            
            else:
                submit_button = st.button("🔍 Get Answer", type="primary")
                
                if (submit_button or selected_query) and user_input.strip():
                    with st.spinner("🤖 Agent is retrieving context and refining final response..."):
                        try:
                            # Invoke Agentic RAG
                            res = agentic_rag_model(
                                query=user_input,
                                retriever=rag_system["retriever"],
                                llm=rag_system["llm"],
                                top_k=top_k,
                                min_score=min_score
                            )
                            
                            answer = res["answer"]
                            sources = res["sources"]
                            
                            # Render Answer
                            st.markdown("#### 🤖 Final Answer")
                            if answer == "NOT FOUND IN DOCUMENTS":
                                st.error("❌ **Not Found In Documents**: The agent strictly analyzed the loaded corpus and found no matching factual grounding to support an answer.")
                            else:
                                st.markdown(f"<div class='status-card'>{answer}</div>", unsafe_allow_html=True)
                                
                                # Render Quick Attribution
                                if sources:
                                    st.markdown("---")
                                    st.markdown("**🔍 Primary Sources Utilized:**")
                                    unique_sources = set(r["metadata"].get("source_file", "Unknown") for r in sources)
                                    for s in unique_sources:
                                        st.markdown(f"<span class='source-badge'>📁 {s}</span>", unsafe_allow_html=True)
                                        
                        except Exception as ex:
                            st.error(f"⚠️ Execution Error: {str(ex)}")
                            
    # ------------------ TAB 2: DOCUMENT CORPUS ------------------
    with tab_corpus:
        st.markdown("### 📁 Manage Document Knowledge Corpus")
        
        col_list, col_upload = st.columns([1, 1])
        
        with col_upload:
            if dataset_source == "📁 Upload Custom Files":
                st.markdown("#### 📤 Upload Documents")
                uploaded_files = st.file_uploader(
                    "Drop documents here (PDF, DOCX, TXT, CSV, XML):",
                    type=["pdf", "docx", "txt", "csv", "xml"],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    any_saved = False
                    for uf in uploaded_files:
                        target_path = os.path.join(UPLOAD_DIR, uf.name)
                        # Save if not already existing
                        if not os.path.exists(target_path):
                            with open(target_path, "wb") as f:
                                f.write(uf.getbuffer())
                            any_saved = True
                    
                    if any_saved:
                        with st.spinner("⚡ Saving files and re-indexing knowledge corpus..."):
                            st.success("✅ Files saved successfully! Re-indexing database...")
                            st.cache_resource.clear()
                            st.rerun()
            else:
                st.markdown("#### 💡 Preloaded Demo Active")
                st.info("The application is in demo mode using the `agentic_rag_guide.txt` file. To query your own documents, select **📁 Upload Custom Files** at the top of the dashboard.")
                st.markdown("##### Demo File Context Preview:")
                demo_path = os.path.join(DEMO_DIR, "agentic_rag_guide.txt")
                if os.path.exists(demo_path):
                    with open(demo_path, "r", encoding="utf-8") as df_r:
                        st.text_area("File preview:", df_r.read()[:800] + "...\n[Truncated for UI preview]", height=200, disabled=True)
                        
        with col_list:
            st.markdown("#### 📄 Current Ingested Files")
            files = os.listdir(UPLOAD_DIR)
            if not files:
                st.info("No files currently inside the retrieval directory.")
            else:
                for idx, f in enumerate(files):
                    f_path = os.path.join(UPLOAD_DIR, f)
                    if os.path.exists(f_path):
                        size_kb = os.path.getsize(f_path) / 1024.0
                        st.markdown(f"**{idx + 1}. {f}** `({size_kb:.2f} KB)`")
                
                st.markdown("---")
                if rag_system:
                    st.metric("Total Loaded Pages/Docs", rag_system["docs_count"])
                    st.metric("Total Extracted Vector Chunks", rag_system["chunks_count"])
                    
                # Reset Option
                st.markdown("#### 🗑️ Reset Database")
                if st.button("⚠️ Wipe Vector Store and Uploads", type="secondary"):
                    with st.spinner("🗑️ Clearing vector store database and uploaded cache..."):
                        # Wipe Uploads
                        for f in os.listdir(UPLOAD_DIR):
                            try:
                                os.remove(os.path.join(UPLOAD_DIR, f))
                            except Exception:
                                pass
                        # Wipe ChromaDB directory
                        if os.path.exists(VECTOR_DIR):
                            try:
                                shutil.rmtree(VECTOR_DIR)
                            except Exception:
                                pass
                        st.cache_resource.clear()
                        st.success("✅ Vector Store database and uploaded cache cleared successfully!")
                        st.rerun()

    # ------------------ TAB 3: RETRIEVAL INSPECTOR ------------------
    with tab_inspector:
        st.markdown("### 🔍 Real-Time Semantic Retrieval Inspector")
        st.markdown("This developer inspection page visualizes the exact vector chunks retrieved from ChromaDB, showing raw content, metadata, and calculated cosine similarity confidence levels.")
        
        inspector_query = st.text_input(
            "Enter a search query to test retrieval:",
            placeholder="Type search terms here to inspect raw chunks..."
        )
        
        if inspector_query.strip():
            if not rag_system:
                st.warning("RAG system not initialized. Ensure documents and keys are loaded.")
            else:
                with st.spinner("Retrieving vector matches..."):
                    results = rag_system["retriever"].retrieve(inspector_query, top_k=top_k)
                    
                    if not results:
                        st.info("No chunks matched the search criteria.")
                    else:
                        st.markdown(f"#### Retrieved Top-{len(results)} Chunks:")
                        for idx, r in enumerate(results):
                            dist = r.get("distance", 1.0)
                            sim = 1.0 / (1.0 + dist)
                            meta = r.get("metadata", {})
                            source = meta.get("source_file", "Unknown File")
                            
                            st.markdown(f"""
                            <div class="status-card">
                                <div>
                                    <span class="source-badge">Chunk #{idx + 1}</span>
                                    <span class="source-badge" style="background-color: #7c3aed;">📁 {source}</span>
                                    <span class="score-badge">Similarity Match Score: {sim:.4f}</span>
                                </div>
                                <div style="margin-top: 0.8rem; font-family: monospace; font-size: 0.9rem; padding: 0.8rem; background-color: rgba(0,0,0,0.03); border-radius: 6px;">
                                    {r['content']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

# 7. Footnote & Educational Center (Conceptual Explainer with Resources)
st.markdown("---")
col_footer_1, col_footer_2 = st.columns([2, 1])
with col_footer_1:
    st.markdown("""
    ### 📖 How this RAG Works
    Retrieval-Augmented Generation (RAG) is a state-of-the-art AI paradigm designed to anchor Large Language Models in ground-truth facts. It works by combining a custom document database search with smart generative validation:
    1. **Semantic Ingestion**: Files are read, parsed, and carefully divided into small, overlapping chunks to retain local context.
    2. **Vector Space Indexing**: Chunks are processed by a semantic embedding model (`all-MiniLM-L6-v2`) that outputs 384-dimensional dense vectors stored in a localized, high-speed **ChromaDB**.
    3. **Semantic Retrieval**: The search query is vectorized, and **ChromaDB** does a cosine-similarity retrieval to find the most relevant chunks.
    4. **Agentic Self-Correction**: The Agent passes the retrieved knowledge chunks into Groq's high-speed inference pipeline with strict grounding prompt validators to block hallucinations.
    """)
with col_footer_2:
    st.markdown("""
    ### 🔗 Existing Educational Web Pages & Guides
    * [LangChain RAG Tutorial & Guide](https://python.langchain.com/docs/tutorials/rag/) — Comprehensive walkthrough explaining the core concepts of RAG pipelines.
    * [IBM Watson: What is Retrieval-Augmented Generation?](https://www.ibm.com/topics/retrieval-augmented-generation) — High-level educational introduction to RAG.
    * [ChromaDB Vector Store Quickstart](https://docs.trychroma.com/) — Official reference for ChromaDB usage and HNSW dense vector space matching.
    * [Groq Cloud Console Guide](https://console.groq.com/docs/quickstart) — High-speed LLM execution reference page.
    """)
