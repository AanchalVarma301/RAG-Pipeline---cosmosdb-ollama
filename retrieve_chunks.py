
import os
import numpy as np
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from sentence_transformers import SentenceTransformer

# 1. Load Cosmos DB configuration
load_dotenv()

endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE")
container_name = os.getenv("COSMOS_CONTAINER")

if not all([endpoint, key, database_name, container_name]):
    raise ValueError("Check the Cosmos DB settings in your .env file.")

# 2. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. Connect to Cosmos DB Emulator
client = CosmosClient(
    url=endpoint,
    credential=key,
    connection_verify=False
)

try:
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # 4. Retrieve stored chunks
    query = """
    SELECT c.id, c.document_name, c.chunk_number, c.text, c.embedding
    FROM c
    """

    items = list(
        container.query_items(
            query=query,
            enable_cross_partition_query=True
        )
    )

    print(f"Records retrieved from Cosmos DB: {len(items)}")

    if not items:
        raise ValueError("No records found in Cosmos DB.")

    # 5. Ask a question
    question = input("\nAsk a question about the PDFs: ").strip()

    if not question:
        raise ValueError("Please enter a question.")

    # 6. Generate an embedding for the question
    question_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    # 7. Calculate similarity with valid stored embeddings
    results = []
    skipped_count = 0

    for item in items:
        # Skip records without an embedding
        if not item.get("embedding"):
            skipped_count += 1
            continue

        stored_vector = np.array(
            item["embedding"],
            dtype=np.float32
        )

        # Skip vectors with incorrect dimensions
        if stored_vector.shape != question_vector.shape:
            print(f"Skipping invalid embedding: {item.get('id', 'unknown')}")
            skipped_count += 1
            continue

        # Normalized vectors: dot product equals cosine similarity
        similarity = float(np.dot(question_vector, stored_vector))

        results.append({
            "document_name": item.get("document_name", "Unknown"),
            "chunk_number": item.get("chunk_number", "Unknown"),
            "text": item.get("text", ""),
            "similarity": similarity
        })

    print(f"Valid chunks used: {len(results)}")
    print(f"Records skipped: {skipped_count}")

    if not results:
        raise ValueError(
            "No valid embedded chunks found. "
            "Check your Cosmos DB records."
        )

    # 8. Sort by similarity and select the top 3
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    top_results = results[:3]

    # 9. Display the most relevant passages
    print("\nMost relevant passages:\n")

    for rank, result in enumerate(top_results, start=1):
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"Document: {result['document_name']}")
        print(f"Chunk: {result['chunk_number']}")
        print(f"Similarity score: {result['similarity']:.4f}")
        print("\nText:")
        print(result["text"])
        print()

finally:
    client.close()