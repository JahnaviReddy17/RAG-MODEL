# pipeline/agent/agentic.py

def agentic_rag_model(query, retriever, llm, min_score=0.2):
    # 2️⃣ Retrieve
    results = retriever.retrieve(query, top_k=7)
    if not results:
        return "NOT FOUND IN DOCUMENTS"

    # 3️⃣ Build context
    context = "\n\n".join([r["content"] for r in results])

    # 4️⃣ Strict prompt (NO hallucination allowed)
    prompt = f"""
RULES:
- Answer ONLY using the context below.
- You MAY summarize or combine information from multiple parts of the context.
- Do NOT use outside knowledge.
- The document may describe a person (candidate/applicant/individual).
- If the requested information is not present in the context, reply EXACTLY:
  NOT FOUND IN DOCUMENTS

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    answer = llm.invoke(prompt).strip()

    # 5️⃣ Final safety net
    if not answer or "NOT FOUND" in answer.upper():
        return "NOT FOUND IN DOCUMENTS"

    return answer
