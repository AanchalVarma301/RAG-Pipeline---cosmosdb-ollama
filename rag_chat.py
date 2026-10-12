
import os
import requests
import numpy as np

from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Load Cosmos DB configuration from .env
# --------------------------------------------------

load_dotenv()

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")

# Ollama model already installed on your computer
OLLAMA_MODEL = "mistral:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --------------------------------------------------
# 2. Connect to the local Cosmos DB Emulator
# --------------------------------------------------

client = CosmosClient(
    COSMOS_ENDPOINT,
    credential=COSMOS_KEY,
    connection_verify=False
)

database = client.get_database_client(COSMOS_DATABASE)
container = database.get_container_client(COSMOS_CONTAINER)

# --------------------------------------------------
# 3. Load the same embedding model used during ingestion
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 4. Retrieve the most relevant chunks from Cosmos DB
# --------------------------------------------------

def retrieve_relevant_chunks(question, top_k=3):

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    query = """
    SELECT c.id, c.document_name, c.chunk_number,
           c.text, c.embedding
    FROM c
    """

    items = list(
        container.query_items(
            query=query,
            enable_cross_partition_query=True
        )
    )

    results = []

    for item in items:
        embedding = item.get("embedding")

        # Skip records without embeddings
        if not isinstance(embedding, list):
            continue

        if len(embedding) != len(question_embedding):
            continue

        # Both vectors are normalized, so dot product
        # calculates cosine similarity.
        score = float(
            np.dot(question_embedding, np.array(embedding))
        )

        results.append({
            "document_name": item.get("document_name", "Unknown"),
            "chunk_number": item.get("chunk_number", "Unknown"),
            "text": item.get("text", ""),
            "score": score
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 5. Send the question and retrieved passages to Mistral
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I couldn't find any relevant passages in the documents."

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
Passage {index}
Source: {chunk['document_name']}
Chunk: {chunk['chunk_number']}
Text:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions about company policies.

Answer the user's question using only the supplied document passages.

Rules:
1. Do not invent facts that are not in the passages.
2. If the passages do not contain enough information, say so.
3. Give a clear, direct answer in simple language.
4. After the answer, list the source document names and chunk numbers.

Document passages:
{context}

User question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# --------------------------------------------------
# 6. Run the complete RAG pipeline
# --------------------------------------------------

def main():

    print("\nRAG application is ready!")
    print("Ask a question about the Infosys and Accenture PDFs.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("RAG application stopped.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            # Retrieval happens first
            chunks = retrieve_relevant_chunks(question)

            if not chunks:
                print("\nNo usable document chunks were found.\n")
                continue

            print("\n--- Retrieved passages ---")

            for index, chunk in enumerate(chunks, start=1):
                print(f"\nPassage {index}")
                print(f"Source: {chunk['document_name']}")
                print(f"Chunk: {chunk['chunk_number']}")
                print(f"Similarity: {chunk['score']:.4f}")
                print(chunk["text"][:500])

            # Mistral generates an answer using those passages
            print("\n--- Mistral's answer ---")

            answer = generate_answer(question, chunks)
            print(answer)
            print()

        except requests.RequestException as error:
            print("\nCould not contact Ollama.")
            print("Check that Ollama is running and the model is available.")
            print(f"Details: {error}\n")

        except Exception as error:
            print("\nAn error occurred:")
            print(error)


if __name__ == "__main__":
    main()