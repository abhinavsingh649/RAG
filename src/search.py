import os
from dotenv import load_dotenv
from src.vectorstore import FAISSVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = 'faiss_store', embedding_model: str = "all-MiniLM-L6-v2", llm_model = "gemini-3.5-flash"):
        self.vector_store = FAISSVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, 'faiss.idx')
        meta_path = os.path.join(persist_dir, 'metadata.pkl')
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents('data')
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-3.5-flash",
            temperature=0.7,
            max_tokens=1024
        )
        print(f"[INFO] Gemini LLM Initialized")

    def search_and_summarized(self, query:str, top_k: int = 5):
        results = self.vector_store.query(query, top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

    # Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    
