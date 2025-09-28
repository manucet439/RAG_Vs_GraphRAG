import faiss
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import *

class FAISSIndexer:
    def __init__(self):
        """Initialize FAISS components"""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.documents = None
        
    def load_and_split_documents(self, file_path: str):
        """Load documents from file and split into chunks"""
        print(f"Loading documents from {file_path}")
        
        # Load documents
        raw_documents = TextLoader(file_path, encoding="latin-1").load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,  # Characters instead of tokens
        chunk_overlap=CHUNK_OVERLAP * 2,
        separators=["\n________________________________________\n", "\n\n", "\n", ". ", " "]
        )
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"Loaded {len(documents)} chunks for FAISS indexing.")
        self.documents = documents
        return documents
    
    def create_faiss_index(self, documents):
        """Create FAISS vector index from documents"""
        print("Creating FAISS vector index...")
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print("FAISS index created successfully!")
        return self.vector_store
    
    def build_index(self, file_path: str = SYNTHETIC_DATA_PATH):
        """Complete FAISS indexing pipeline"""
        documents = self.load_and_split_documents(file_path)
        return self.create_faiss_index(documents)
    
    def save_index(self, path: str = "faiss_index"):
        """Save FAISS index to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"FAISS index saved to {path}")
    
    def load_index(self, path: str = "faiss_index"):
        """Load FAISS index from disk"""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"FAISS index loaded from {path}")
            return self.vector_store
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            return None
    
    def get_index_stats(self):
        """Get basic statistics about the FAISS index"""
        if self.vector_store:
            # Get the underlying FAISS index
            faiss_index = self.vector_store.index
            vector_count = faiss_index.ntotal
            dimension = faiss_index.d
            
            print(f"FAISS Index Statistics:")
            print(f"- Vectors: {vector_count}")
            print(f"- Dimensions: {dimension}")
            print(f"- Documents: {len(self.documents) if self.documents else 'N/A'}")
            
            return {
                "vectors": vector_count, 
                "dimensions": dimension,
                "documents": len(self.documents) if self.documents else 0
            }
        else:
            print("FAISS index not initialized")
            return None