import chromadb
import os
import uuid

class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_dir="data/vector_store"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents, embeddings):
        ids, texts, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(str(uuid.uuid4()))
            texts.append(doc.page_content)
            metas.append(doc.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs
        )
