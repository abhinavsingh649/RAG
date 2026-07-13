from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FAISSVectorStore



if __name__ == "__main__":
    # docs = load_all_documents("data")
    store = FAISSVectorStore()
    # store.build_from_documents(docs)
    store.load()
    print(store.query("classical Mechanics", top_k = 3))
     

