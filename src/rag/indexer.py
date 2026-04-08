"""
RAG Indexer — converts ad campaign data into a searchable vector index.

What this module does:
- Loads feature-engineered data (features.csv)
- Converts each row into a human-readable text document
- Generates embeddings (vector representations) for each document
- Stores everything in ChromaDB (a vector database)
- Provides a query function to find relevant rows by natural language

What is a vector database?
Imagine a library where books aren't sorted alphabetically,
but by MEANING. If you ask "books about space travel",
it finds sci-fi novels even if their titles don't mention "space".

How it works:
1. Each data row → text description ("facebook_tier1 on 2024-01-15: CTR=0.04...")
2. Text → embedding vector (a list of 384 numbers that capture meaning)
3. Vectors stored in ChromaDB
4. Question → embedding → find closest vectors → return matching rows

Input:  data/processed/features.csv
Output: chroma_db/ folder (persistent vector database)
"""

import pandas as pd
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS AND CONFIG
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
CHROMA_PATH = Path("chroma_db")

# Embedding model: all-MiniLM-L6-v2
# - Runs locally (no API key needed)
# - Fast and lightweight (80MB)
# - Produces 384-dimensional vectors
# - Good quality for semantic search
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection name (like a "table" in a regular database)
COLLECTION_NAME = "ad_campaigns"


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    """Load the feature-engineered dataset."""
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows from {FEATURES_PATH}")
    return df


# ============================================================
# CONVERT ROWS TO TEXT DOCUMENTS
# ============================================================

def row_to_document(row):
    """
    Convert a single DataFrame row into a human-readable text document.

    Why text?
    Embedding models understand TEXT, not numbers.
    We need to describe each row in words so the model can
    understand what it's about and find it when asked.

    Example output:
    "Campaign facebook_tier1 (social) on 2024-01-15:
     Impressions: 1500, Clicks: 60, Leads: 12, Orders: 6.
     Spent: $150.00, Revenue: $450.00.
     CTR: 4.00%, CPC: $2.50, ROAS: 3.00, CPL: $12.50.
     Conversion rate: 10.00%.
     ROAS change week-over-week: -15.0%.
     CTR change week-over-week: +5.2%.
     This campaign is flagged as an ANOMALY."

    Parameters:
        row: a single row from the DataFrame (pd.Series)

    Returns:
        string: human-readable description of the row
    """
    # Start with campaign identity
    lines = [
        f"Campaign {row.get('campaign_name', 'unknown')} "
        f"({row.get('category', 'unknown')}) "
        f"on {row.get('c_date', 'unknown')}:"
    ]

    # Raw metrics (if available)
    if "impressions" in row.index:
        lines.append(
            f"Impressions: {int(row['impressions'])}, "
            f"Clicks: {int(row['clicks'])}, "
            f"Leads: {int(row['leads'])}, "
            f"Orders: {int(row['orders'])}."
        )

    if "mark_spent" in row.index:
        lines.append(
            f"Spent: ${row['mark_spent']:.2f}, "
            f"Revenue: ${row['revenue']:.2f}."
        )

    # Computed metrics
    if "ctr" in row.index:
        lines.append(
            f"CTR: {row['ctr'] * 100:.2f}%, "
            f"CPC: ${row['cpc']:.2f}, "
            f"ROAS: {row['roas']:.2f}, "
            f"CPL: ${row['cpl']:.2f}."
        )

    if "conversion_rate" in row.index:
        lines.append(f"Conversion rate: {row['conversion_rate'] * 100:.2f}%.")

    # Week-over-week changes
    if "roas_wow" in row.index and not pd.isna(row["roas_wow"]):
        lines.append(
            f"ROAS change week-over-week: {row['roas_wow'] * 100:+.1f}%."
        )

    if "ctr_wow" in row.index and not pd.isna(row["ctr_wow"]):
        lines.append(
            f"CTR change week-over-week: {row['ctr_wow'] * 100:+.1f}%."
        )

    if "spend_wow" in row.index and not pd.isna(row["spend_wow"]):
        lines.append(
            f"Spend change week-over-week: {row['spend_wow'] * 100:+.1f}%."
        )

    # Anomaly flag
    if "is_anomaly" in row.index:
        if row["is_anomaly"] == 1:
            lines.append("This campaign is flagged as an ANOMALY.")
        else:
            lines.append("This campaign is performing normally.")

    return "\n".join(lines)


def create_documents(df):
    """
    Convert all DataFrame rows into text documents.

    Returns:
        documents: list of text strings
        ids: list of unique IDs (e.g. "row_0", "row_1", ...)
        metadatas: list of dicts with key fields for filtering
    """
    documents = []
    ids = []
    metadatas = []

    for idx, row in df.iterrows():
        doc = row_to_document(row)
        documents.append(doc)
        ids.append(f"row_{idx}")

        # Metadata allows filtering in ChromaDB
        # e.g. "show me only facebook campaigns" or "only anomalies"
        metadata = {
            "campaign_name": str(row.get("campaign_name", "")),
            "category": str(row.get("category", "")),
            "is_anomaly": int(row.get("is_anomaly", 0)),
        }

        if "c_date" in row.index:
            metadata["c_date"] = str(row["c_date"])

        metadatas.append(metadata)

    print(f"Created {len(documents)} documents")
    return documents, ids, metadatas


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Load the sentence-transformers embedding model.

    What is an embedding?
    A way to represent text as a list of numbers (vector).
    Similar texts get similar vectors, so we can find
    "semantically close" documents by comparing vectors.

    Example:
    "CTR dropped sharply" → [0.12, -0.45, 0.78, ...]  (384 numbers)
    "Click rate fell"     → [0.11, -0.43, 0.76, ...]  (very similar!)
    "Revenue increased"   → [-0.32, 0.67, -0.15, ...] (very different)

    The model downloads on first use (~80MB) and caches locally.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded")
    return model


# ============================================================
# CHROMADB INDEX
# ============================================================

def create_index(df):
    """
    Create a ChromaDB collection from the DataFrame.

    Steps:
    1. Convert rows to text documents
    2. Load embedding model
    3. Generate embeddings for all documents
    4. Store in ChromaDB (persistent — saved to disk)

    ChromaDB is like a database, but for vectors:
    - Regular DB: search by exact values (WHERE name = 'facebook')
    - Vector DB: search by MEANING ("campaigns with low performance")

    Parameters:
        df: DataFrame with feature-engineered data

    Returns:
        collection: ChromaDB collection object (ready for queries)
    """
    # Step 1: convert rows to text
    documents, ids, metadatas = create_documents(df)

    # Step 2: load embedding model
    embedding_model = get_embedding_model()

    # Step 3: generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    # Convert to list of lists (ChromaDB format)
    embeddings_list = embeddings.tolist()
    print(f"Generated {len(embeddings_list)} embeddings of dimension {len(embeddings_list[0])}")

    # Step 4: store in ChromaDB
    # PersistentClient saves to disk so we don't rebuild every time
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete old collection if it exists (fresh rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass  # Collection doesn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Ad campaign performance data"},
    )

    # Add all documents with their embeddings and metadata
    collection.add(
        documents=documents,
        embeddings=embeddings_list,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"Index created: {collection.count()} documents in ChromaDB")
    return collection


# ============================================================
# QUERY INDEX
# ============================================================

def get_collection():
    """
    Get an existing ChromaDB collection (without rebuilding).

    Use this when the index is already built and you just
    want to query it.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    return collection


def query_index(question, top_k=5):
    """
    Search the vector index with a natural language question.

    How semantic search works:
    1. Your question → embedding vector
    2. ChromaDB compares this vector with all stored vectors
    3. Returns the top_k most similar documents

    "Similar" means semantically close:
    - "worst performing campaign" finds rows with low ROAS
    - "anomalies in social campaigns" finds flagged social rows

    Parameters:
        question: natural language query string
        top_k: how many results to return (default 5)

    Returns:
        dict with keys:
        - documents: list of text documents
        - metadatas: list of metadata dicts
        - distances: list of similarity scores (lower = more similar)
        - ids: list of document IDs
    """
    # Load embedding model and collection
    embedding_model = get_embedding_model()
    collection = get_collection()

    # Generate embedding for the question
    question_embedding = embedding_model.encode([question]).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    print(f"Found {len(results['documents'][0])} results for: '{question}'")
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
        "ids": results["ids"][0],
    }


# ============================================================
# MAIN
# ============================================================

def run_indexer():
    """
    Full indexing pipeline:
    1. Load data
    2. Create documents
    3. Build vector index
    4. Test with a sample query
    """
    # Step 1: load data
    df = load_features()

    # Step 2: build index
    collection = create_index(df)

    # Step 3: test query
    print("\n" + "=" * 60)
    print("  TEST QUERIES")
    print("=" * 60)

    test_questions = [
        "Which campaigns have the worst ROAS?",
        "Show me anomalies in social campaigns",
        "Which campaign spent the most money?",
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        results = query_index(question, top_k=3)
        for i, (doc, dist) in enumerate(zip(results["documents"], results["distances"])):
            # Show first line of each result + similarity score
            first_line = doc.split("\n")[0]
            print(f"  {i+1}. [{dist:.4f}] {first_line}")

    print("\n" + "*" * 60)
    print("  Indexer complete!")
    print("*" * 60)


if __name__ == "__main__":
    run_indexer()