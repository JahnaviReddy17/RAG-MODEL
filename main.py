# main.py
import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

def print_banner():
    banner = """
====================================================================
               🤖 AGENTIC RAG SYSTEM ORCHESTRATOR
====================================================================
A highly modular, secure, and strict source-grounded RAG pipeline.

   [Core Pipeline Architecture]
   📁 Ingestion (pdf_loader) ────> ✂️ Processing (text_splitter)
                                            │
   💬 Agentic LLM (Llama-3)   <──── 🔍 Retrieval (VectorStore)
====================================================================
"""
    print(banner)

def check_environment():
    print("📋 Diagnostic Status Checks:")
    
    # Check GROQ API Key
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        masked_key = groq_key[:6] + "..." + groq_key[-4:] if len(groq_key) > 10 else "configured"
        print(f"  ✅ Groq API Key: Configured ({masked_key})")
    else:
        print("  ⚠️  Groq API Key: MISSING (Please configure GROQ_API_KEY in your .env file or input in the UI sidebar)")

    # Check directories
    upload_dir = "data/uploaded_pdfs"
    demo_dir = "data/demo_files"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)
    
    uploaded_files = os.listdir(upload_dir)
    print(f"  📁 Ingested Corpus Directory: '{upload_dir}' ({len(uploaded_files)} files loaded)")
    for idx, f in enumerate(uploaded_files[:5]):
        print(f"     - {f}")
    if len(uploaded_files) > 5:
        print(f"     - ...and {len(uploaded_files) - 5} more files")
        
    print("====================================================================\n")

def run_streamlit_ui():
    print("🚀 Starting Streamlit Web Application...")
    ui_path = "ui.py"
    if not os.path.exists(ui_path):
        print(f"❌ Error: Could not locate user interface file '{ui_path}' in workspace root.")
        return
        
    try:
        # Run streamlit run ui.py using subprocess
        subprocess.run(["streamlit", "run", ui_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web Application stopped successfully.")
    except Exception as e:
        print(f"❌ Error running Streamlit server: {str(e)}")
        print("💡 Hint: Ensure streamlit is installed by running 'pip install -r requirements.txt'")

def run_cli_chat():
    print("💬 Entering Interactive CLI Chat Mode...")
    
    # Delay imports to avoid slow startup and let them fail cleanly if dependencies are missing
    try:
        from pipeline.ingestion.pdf_loader import process_all_docs
        from pipeline.processing.text_splitter import split_documents
        from pipeline.embeddings.embedding_manager import EmbeddingManager
        from pipeline.vectorstore.chroma_store import VectorStore
        from pipeline.retrieval.retriever import RAGRetriever
        from pipeline.llm.groq_llm import GroqLLM
        from pipeline.agent.agentic import agentic_rag_model
    except ImportError as e:
        print(f"❌ Dependency Error: {str(e)}")
        print("💡 Hint: Please install all dependencies with: pip install -r requirements.txt")
        return

    # Check key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: Groq API Key must be set in your environment or '.env' file to use the CLI.")
        return

    # Validate that we have documents to search
    upload_dir = "data/uploaded_pdfs"
    if not os.listdir(upload_dir):
        print(f"⚠️ Warning: Knowledge corpus is empty. Copying preloaded demo file...")
        demo_src = "data/demo_files/agentic_rag_guide.txt"
        demo_dst = os.path.join(upload_dir, "agentic_rag_guide.txt")
        if os.path.exists(demo_src):
            import shutil
            shutil.copy(demo_src, demo_dst)
            print(f"✅ Preloaded demo document successfully copied to '{upload_dir}'.")
        else:
            print("❌ Error: No documents found and demo file is missing. Please add files to 'data/uploaded_pdfs'.")
            return

    # Initialize RAG components
    print("⚡ Initializing vector store and embedding system (please wait)...")
    try:
        docs = process_all_docs(upload_dir)
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        
        embedder = EmbeddingManager()
        embeddings = embedder.generate_embeddings([c.page_content for c in chunks])
        
        store = VectorStore()
        store.add_documents(chunks, embeddings)
        
        retriever = RAGRetriever(store, embedder)
        llm = GroqLLM()
        print("✅ Pipeline initialized successfully. Ask your questions below!")
    except Exception as err:
        print(f"❌ Initialization Error: {str(err)}")
        return

    print("\n--------------------------------------------------------------------")
    print("Commands:")
    print("  'exit' or 'quit' - Leave the chat session")
    print("  'files'          - List currently loaded source documents")
    print("--------------------------------------------------------------------")
    
    while True:
        try:
            query = input("\n👤 Ask a Question > ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                print("👋 Exiting RAG CLI session. Goodbye!")
                break
            if query.lower() == 'files':
                print(f"Ingested documents: {', '.join(os.listdir(upload_dir))}")
                continue

            print("🤖 Thinking...")
            res = agentic_rag_model(query, retriever, llm)
            
            print(f"\n🤖 Answer:\n{res['answer']}")
            
            if res['sources']:
                print("\n🔍 Used Sources:")
                unique_sources = set(r["metadata"].get("source_file", "Unknown") for r in res["sources"])
                for s in unique_sources:
                    print(f"  • {s}")
                    
        except KeyboardInterrupt:
            print("\n👋 Chat session stopped.")
            break
        except Exception as e:
            print(f"⚠️ Error answering query: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG System Orchestrator Dashboard")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit Web UI dashboard")
    parser.add_argument("--cli", action="store_true", help="Launch the interactive CLI chat session")
    args = parser.parse_args()

    print_banner()
    check_environment()

    # Determine command or open default selection menu
    if args.ui:
        run_streamlit_ui()
    elif args.cli:
        run_cli_chat()
    else:
        print("💡 No interface flag was provided. Please choose an execution mode:")
        print("  [1] Launch Modern Streamlit Web UI (Default)")
        print("  [2] Start Terminal-based Interactive CLI Chat")
        print("  [3] Exit Orchestrator")
        
        try:
            choice = input("\nEnter choice (1, 2, or 3) > ").strip()
            if choice == "1" or choice == "":
                run_streamlit_ui()
            elif choice == "2":
                run_cli_chat()
            else:
                print("👋 Goodbye!")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
