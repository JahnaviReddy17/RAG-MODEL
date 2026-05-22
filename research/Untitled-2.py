# %%
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# %% [markdown]
# ## Embeddings

# %%
### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
all_pdf_documents = process_all_pdfs("./data/pdf_files")

#all_pdf_documents

# %%
### Text splitting get into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

chunks=split_documents(all_pdf_documents)
#chunks

# %%
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# %%
class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        
        """Generate embeddings for a list of texts
        Args:texts: List of text strings to embed
        Returns:numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


## initialize the embedding manager

embedding_manager=EmbeddingManager()
embedding_manager

# %% [markdown]
# ## VECTOR STORE

# %%
import os
import uuid
import numpy as np
import chromadb
from typing import List, Any


# %%
class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore=VectorStore()
vectorstore

# %%
chunks

# %%
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

##store int he vector dtaabase
vectorstore.add_documents(chunks,embeddings)

# %% [markdown]
# ## RETRIEVAL

# %%
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 / (1 + distance)

                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)

rag_retriever.retrieve("Explain the applications of nanotechnology in Applied Physics.")

# %%
import os
from dotenv import load_dotenv
load_dotenv()


# %%
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# %% [markdown]
# ## AGENTIC AI

# %%
class AgenticQueryAnalyzer:
    """Always retrieve for every query."""
    def __init__(self):
        pass

    def should_retrieve(self, query: str) -> bool:
        return True

class AgenticRetrievalController:
    def __init__(self, top_k=7):
        self.top_k = top_k

    def choose_top_k(self, query: str) -> int:
        return self.top_k

class AgenticAnswerRefiner:
    """
    Guarantees NO hallucination.
    If answer not fully supported by context → return "NOT FOUND".
    """
    def __init__(self, llm):
        self.llm = llm

    def refine(self, query, answer, context):
        prompt = f"""
STRICT RULES:
- You MUST answer ONLY using the context below.
- If the answer is not fully present in the context → reply EXACTLY:
  NOT FOUND IN DOCUMENTS
- Do NOT add, guess, assume, or generate anything outside the context.
- You may rephrase, but do NOT add new facts.

QUESTION:
{query}

ANSWER GIVEN:
{answer}

CONTEXT (ONLY source of truth):
{context[:2000]}

Is the given answer fully supported by context?

Reply with ONLY ONE of:
1. OK
2. NOT FOUND IN DOCUMENTS
"""

        result = self.llm.invoke([prompt]).content.strip()

        if result == "OK":
            return answer

        return "The requested information is not present in the provided documents."



# %%
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class GroqLLM:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        print(f"✅ Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful AI assistant. Use the following context to answer the question accurately.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )

        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2,
                max_completion_tokens=max_length  # ✅ correct param name for Groq SDK
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        simple_prompt = f"Based on this context: {context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0.2
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error: {str(e)}"



# %% [markdown]
# ## RAG_MODEL

# %%
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize Groq client (no langchain_groq needed)
client = Groq(api_key=groq_api_key)
MODEL_NAME = "llama-3.1-8b-instant"   # active model (as of Nov 2025)

# --- Simple ChatGroq-like wrapper (drop-in replacement)
class ChatGroq:
    def __init__(self, groq_api_key, model_name, temperature=0.1, max_tokens=1024):
        self.client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages):
        # messages is a list of dicts or objects with .content
        user_text = ""
        if isinstance(messages, list):
            for m in messages:
                user_text += getattr(m, "content", str(m)) + "\n"
        else:
            user_text = str(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_text}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        return type("Response", (), {"content": response.choices[0].message.content.strip()})

# ✅ use our wrapper instead of the broken langchain_groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=MODEL_NAME,
    temperature=0.1,
    max_tokens=1024
)
query_analyzer = AgenticQueryAnalyzer()
retrieval_controller = AgenticRetrievalController()
answer_refiner = AgenticAnswerRefiner(llm)
def agentic_rag_model(
    query,
    retriever,
    llm,
    query_analyzer,
    retrieval_controller,
    answer_refiner,
    min_score=0.2,
):
    # 1. Decide whether retrieval is needed
    must_retrieve = query_analyzer.should_retrieve(query)

    if not must_retrieve:
        print("Agent: No retrieval needed.")
        direct = llm.invoke([query]).content
        return {"answer": [direct], "sources": [], "confidence": 1.0}

    # 2. Choose top_k
    top_k = retrieval_controller.choose_top_k(query)
    print(f"Agent selected top_k = {top_k}")

    # 3. Retrieve
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {'answer': ['No relevant context found.'], 'sources': []}

    context = "\n\n".join([doc["content"] for doc in results])
    sources = [{
        "source": doc["metadata"].get("source_file", "unknown"),
        "page": doc["metadata"].get("page", None),
        "score": doc["similarity_score"],
        "preview": doc["content"][:150]
    } for doc in results]

    # 4. LLM first answer
    prompt = f"""Use the context to answer the question.

Context:
{context}

Question: {query}
Answer:
"""

    initial_answer = llm.invoke([prompt]).content

    # 5. Refine the answer
    refined_answer = answer_refiner.refine(query, initial_answer, context)

    confidence = max([doc["similarity_score"] for doc in results])

    return {
        "answer": [refined_answer],
        "sources": sources,
        "confidence": confidence,
        "context": context
    }
def agentic_rag_model(
    query,
    retriever,
    llm,
    query_analyzer,
    retrieval_controller,
    answer_refiner,
    min_score=0.2,
):
    # Always retrieve (Analyzer is simple)
    _ = query_analyzer.should_retrieve(query)

    # How many chunks to retrieve
    top_k = retrieval_controller.choose_top_k(query)

    # Retrieve
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            'answer': ["The requested information is not present in the provided documents."],
            'sources': [],
            'confidence': 0.0
        }

    # Build context
    context = "\n\n".join([doc["content"] for doc in results])

    # SOURCES
    sources = [{
        "source": doc["metadata"].get("source_file", "unknown"),
        "page": doc["metadata"].get("page", None),
        "score": doc["similarity_score"],
        "preview": doc["content"][:150]
    } for doc in results]

    # Initial answer
    prompt = f"""
Answer ONLY using the context below.
If the answer is not present → reply "NOT FOUND IN DOCUMENTS".

CONTEXT:
{context}

Question: {query}
Answer:
"""

    initial_answer = llm.invoke([prompt]).content.strip()

    # Refine (hallucination block)
    refined = answer_refiner.refine(query, initial_answer, context)

    confidence = max([doc["similarity_score"] for doc in results])

    return {
        "answer": [refined],
        "sources": sources,
        "confidence": confidence,
        "context": context
    }


# %%
answer = agentic_rag_model(
    'What are you doing?',
    rag_retriever,
    llm,
    query_analyzer,
    retrieval_controller,
    answer_refiner
)

for line in answer['answer']:
    print(line)



