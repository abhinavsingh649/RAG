from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FAISSVectorStore
from src.search import RAGSearch


if __name__ == "__main__":
    # docs = load_all_documents("data")
    store = FAISSVectorStore()
    # store.build_from_documents(docs)
    store.load()
    # print(store.query("classical Mechanics", top_k = 3))
    
    rag_search = RAGSearch()
    query = "Classical Mechanics"
    summary = rag_search.search_and_summarized(query, top_k = 3)
    print("Summary :", summary)


