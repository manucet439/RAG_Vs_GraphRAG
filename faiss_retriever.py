from typing import List, Dict, Any

class FAISSRetriever:
    def __init__(self, vector_store):
        """Initialize with FAISS vector store"""
        self.vector_store = vector_store
    
    def retrieve(self, question: str, k: int = 4) -> str:
        """Main retrieval method using vector similarity search"""
        print(f"FAISS Search query: {question}")
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(question, k=k)
        
        # Combine document content
        retrieved_content = []
        for i, doc in enumerate(docs):
            retrieved_content.append(f"Document {i+1}:\n{doc.page_content}")
        
        final_data = "\n\n".join(retrieved_content)
        
        print(f"\nFAISS Retrieval Result:")
        print(f"Document chunks found: {len(docs)}")
        
        return final_data
    
    def retrieve_with_scores(self, question: str, k: int = 4) -> List[tuple]:
        """Retrieve documents with similarity scores"""
        print(f"FAISS Search with scores: {question}")
        
        # Perform similarity search with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        print(f"Retrieved {len(docs_with_scores)} documents with scores")
        
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Document {i+1} (Score: {score:.4f}): {doc.page_content[:100]}...")
        
        return docs_with_scores
    
    def retrieve_formatted(self, question: str, k: int = 4) -> str:
        """Retrieve and format results with scores for comparison"""
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        formatted_results = []
        for i, (doc, score) in enumerate(docs_with_scores):
            formatted_results.append(
                f"Document {i+1} (Similarity Score: {score:.4f}):\n{doc.page_content}\n"
            )
        
        return "\n".join(formatted_results)
    
    def get_most_relevant_chunks(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Get most relevant chunks with metadata"""
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        results = {
            "query": question,
            "num_results": len(docs_with_scores),
            "documents": []
        }
        
        for i, (doc, score) in enumerate(docs_with_scores):
            doc_info = {
                "rank": i + 1,
                "similarity_score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            results["documents"].append(doc_info)
        
        return results