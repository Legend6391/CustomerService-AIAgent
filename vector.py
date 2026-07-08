import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore

# Initialize local disk cache directory
cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "embeddings")
os.makedirs(cache_dir, exist_ok=True)
store = LocalFileStore(cache_dir)

# Load CSV data
df = pd.read_csv("insurance.csv")

base_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    keep_alive=300,
    base_url="http://localhost:11434"  # Windows/Mac: connects to host
)

safe_namespace = base_embeddings.model.replace(":", "_")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=store,
    namespace=safe_namespace,
    query_embedding_cache=True
)

db_location = "chrome_langchain_db"

vector_store = Chroma(
    collection_name="insurance",
    embedding_function=cached_embeddings,
    persist_directory=db_location
)

# Check count to see if we need to reload/rebuild
num_rows = len(df)
try:
    db_count = vector_store._collection.count()
except Exception as e:
    db_count = 0

if db_count != num_rows:
    print(f"Database count ({db_count}) mismatch with CSV rows ({num_rows}). Rebuilding collection...")
    try:
        vector_store.delete_collection()
    except Exception as e:
        print(f"Could not delete collection: {e}")
    
    # Re-instantiate vector store to recreate collection
    vector_store = Chroma(
        collection_name="insurance",
        embedding_function=cached_embeddings,
        persist_directory=db_location
    )
    
    documents = []
    for idx, row in df.iterrows():
        document = Document(
            page_content=f"Question: {row['question']} Answer: {row['answer']}",
            metadata={"source": "insurance.csv", "id": row["id"], "category": row["category"]},
        )
        documents.append(document)
        
    print(f"Indexing {len(documents)} documents in batches...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"Indexed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")
else:
    print(f"Successfully loaded existing database with {db_count} documents.")

def retrieve_with_confidence(query):
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=3
    )
    if not results:
        return "", "General", "", 0.0, [], []

    context = "\n\n".join(
        doc.page_content for doc, _ in results
    )

    best_score = results[0][1]
    best_doc = results[0][0]
    category = best_doc.metadata.get("category", "General")
    best_answer = best_doc.page_content

    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

    return context, category, best_answer, best_score, docs, scores
