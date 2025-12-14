Agentic RAG System (Hallucination-Free, Production-Ready)

A Retrieval-Augmented Generation system with strict grounding, agent-controlled retrieval, and zero-hallucination guarantees.

Built a production-grade Retrieval-Augmented Generation system with explicit agentic control over retrieval and answer validation, ensuring zero hallucinations by enforcing strict document-grounded responses using LLM-based verification.

--->Environment Setup

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

---->How to Run
pip install -r requirements.txt
python main.py


Example usage:

answer = agentic_rag_model(
    "Ask your Question/Query here?",
    rag_retriever,
    llm,
    query_analyzer,
    retrieval_controller,
    answer_refiner
)

print(answer["answer"])
