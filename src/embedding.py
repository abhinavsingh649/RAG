from typing import Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.data_loader import load_all_documents
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Embedding Model = {model_name}")

    def chunk_documents(self, documents: List[Any] ) -> List[Any]:
        Splitter = RecursiveCharacterTextSplitter(
            chunk_overlap = self.chunk_overlap,
            chunk_size = self.chunk_size,
            length_function = len,
            separators=["\n\n","\n", " ", ""],
        )
        
        chunks = Splitter.split_documents(documents)
        print(f"[INFO] split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def embed_documents(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar = True, convert_to_numpy = True)
        print(f"[INFO] Embeddings shape = {embeddings.shape}")
        return embeddings
    
    
        