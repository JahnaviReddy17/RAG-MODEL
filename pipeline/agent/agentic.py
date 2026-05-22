def agentic_rag_model(query, retriever, llm, top_k=5, min_score=0.1):
    # 2️⃣ Retrieve chunks
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {
            "answer": "NOT FOUND IN DOCUMENTS",
            "sources": []
        }

    # Calculate similarity from distance: similarity = 1 / (1 + distance)
    filtered_results = []
    for r in results:
        distance = r.get("distance", 1.0)
        similarity = 1.0 / (1.0 + distance)
        r["similarity"] = similarity
        if similarity >= min_score:
            filtered_results.append(r)

    if not filtered_results:
        return {
            "answer": "NOT FOUND IN DOCUMENTS",
            "sources": []
        }

    # 3️⃣ Build context
    context = "\n\n".join([r["content"] for r in filtered_results])

    # 4️⃣ Strict prompt (NO hallucination allowed)
    prompt = f"""
RULES:
- Answer ONLY using the context below.
- You MAY summarize or combine information from multiple parts of the context.
- Do NOT use outside knowledge.
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
        return {
            "answer": "NOT FOUND IN DOCUMENTS",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": filtered_results
    }
