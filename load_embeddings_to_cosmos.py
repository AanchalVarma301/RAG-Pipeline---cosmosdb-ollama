
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# 1. Load settings from .env
load_dotenv()

endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE")
container_name = os.getenv("COSMOS_CONTAINER")

if not all([endpoint, key, database_name, container_name]):
    raise ValueError("Check the Cosmos DB settings in your .env file.")

# 2. Locate the embedded chunks
input_file = Path("embedded_data") / "embedded_chunks.json"

if not input_file.exists():
    raise FileNotFoundError(f"File not found: {input_file}")

with input_file.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded from JSON: {len(chunks)}")

# 3. Connect to your local Cosmos DB Emulator
# connection_verify=False is for local emulator development only.
client = CosmosClient(
    url=endpoint,
    credential=key,
    connection_verify=False
)

database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# 4. Upload every chunk and its embedding
success_count = 0

for chunk in chunks:
    item = {
        "id": chunk["id"],
        "document_name": chunk["document_name"],
        "chunk_number": chunk["chunk_number"],
        "text": chunk["text"],
        "embedding": chunk["embedding"]
    }

    container.upsert_item(body=item)
    success_count += 1

    if success_count % 10 == 0:
        print(f"Uploaded {success_count}/{len(chunks)} chunks")

print("\nUpload completed successfully!")
print(f"Total chunks uploaded: {success_count}")

# 5. Verify one uploaded document
first_chunk = chunks[0]

saved_item = container.read_item(
    item=first_chunk["id"],
    partition_key=first_chunk["id"]
)

print("\nVerification successful!")
print(f"Document: {saved_item['document_name']}")
print(f"Chunk number: {saved_item['chunk_number']}")
print(f"Embedding dimensions: {len(saved_item['embedding'])}")

client.close()