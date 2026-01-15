DOCSEARCH

TL;DR

Built a RAG-based document question–answering system that retrieves relevant documents for a given query and generates answers strictly from those documents. The system breaks complex questions into sub-questions, applies vector or summary-based retrieval, and aggregates results into a final response. If no relevant documents are found, it returns “not found”, reducing hallucinations and improving reliability.

---------------------------------------------------------------------------------

RAG-Based Document Question–Answering System

This project focuses on building a customized document-based question–answering system using Retrieval-Augmented Generation (RAG) to overcome the limitations of standalone Large Language Models (LLMs). While LLMs excel at language understanding and generation, they often struggle with accuracy, factual grounding, and domain-specific knowledge. To address this, the system integrates external documents through a structured retrieval pipeline, ensuring responses are relevant, reliable, and grounded in source material.

The system answers user queries by retrieving information from a curated document corpus and supplying this retrieved content to the LLM for response generation. For complex queries, the system decomposes the original question into smaller, focused sub-questions. Each sub-question is processed independently using suitable retrieval strategies such as vector-based similarity search or summary-based retrieval. The retrieved information is then used within carefully designed prompt templates to generate accurate answers.

A key aspect of this project is strict document grounding, where the model is instructed to generate answers only from the retrieved documents. If relevant information is not found, the system explicitly indicates that the answer is unavailable, significantly reducing hallucinations. The final response is formed by aggregating answers from individual sub-questions into a single coherent output. This modular pipeline design improves transparency, scalability, and ease of debugging.

The project also explores practical challenges in building real-world RAG systems, including prompt sensitivity, query decomposition accuracy, retrieval performance, and cost–efficiency trade-offs. The implementation demonstrates how minor variations in user queries can affect retrieval results and generation quality, highlighting the importance of robust retrieval tuning and prompt design.

Overall, this project demonstrates how RAG-based document question–answering systems can be used to build reliable, source-grounded AI solutions suitable for applications such as enterprise document search, knowledge extraction, and domain-specific information retrieval.


---------------------------------------------------------------------------------

Implementation Details

1)Document Ingestion:

Load and preprocess documents (PDFs/text).

Split into chunks and generate vector embeddings stored in a vector database.

2)Query Processing:

Decompose complex questions into sub-questions.

Each sub-question is processed independently.

3)Document Retrieval:

Retrieve relevant chunks via vector similarity or summary-based search.

Returns “Not Found” if no relevant documents are found.

4)Answer Generation & Aggregation:

LLM generates answers strictly from retrieved content.

Sub-question answers are combined into a final coherent response.

---------------------------------------------------------------------------------

⚙️ Setup Instructions

# 1. Clone the repository
git clone https://github.com/your-username/RAG-MODEL.git
cd RAG-MODEL

# 2. Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create upload folder (optional, Streamlit does it automatically)
mkdir -p data/uploaded_pdfs

# 5. Run Streamlit app
streamlit run main.py  # assuming your file is main.py

Make sure your document corpus is in the data/ folder (or specify path in the config).