class DocumentRetriever:
    """
    A dense document retriever that encodes documents using SentenceTransformer
    and builds a FAISS index for similarity search.
    """
    def __init__(self, documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.documents = documents
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Encoding %d documents...", len(documents))
        self.embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index = self._build_faiss_index(self.embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dim = embeddings.shape[1]
        logger.info("Building FAISS index with dimension: %d", dim)
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        try:
            logger.info("Encoding query: '%s'", query)
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                results.append((self.documents[idx], float(dist)))
            logger.info("Retrieved %d documents.", len(results))
            return results
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            return []
