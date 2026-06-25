import chromadb


class ChromaStorage:

    def __init__(self, collection="docs"):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection
        )

    def upsert(self, ids, vectors, payloads):

        documents = [
            payload.get("text", "")
            for payload in payloads
        ]

        metadatas = payloads

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_vector, top_k=5):

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        contexts = []
        sources = set()

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):

            contexts.append(doc)

            source = meta.get("source", "")

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }