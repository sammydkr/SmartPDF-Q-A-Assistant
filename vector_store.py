import os
import pickle
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from config import Config

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL
        )
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document], store_name: str = "default"):
        """Create and save a vector store from documents"""
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        save_path = os.path.join(Config.VECTOR_STORE_PATH, store_name)
        self.vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
        
        return self.vector_store
    
    def load_vector_store(self, store_name: str = "default"):
        """Load an existing vector store"""
        load_path = os.path.join(Config.VECTOR_STORE_PATH, store_name)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No vector store found at {load_path}")
        
        self.vector_store = FAISS.load_local(
            load_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded vector store from {load_path}")
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_scores(self, query: str, k: int = 4) -> List[Dict]:
        """Search with similarity scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return results
