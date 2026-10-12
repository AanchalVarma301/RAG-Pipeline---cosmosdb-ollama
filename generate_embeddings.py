# For this first project, we'll use the free, locally runnable model all-MiniLM-L6-v2. 
# It converts text into a 384-number vector and doesn't require an Azure API key. 
# The model downloads the first time you run it

from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

# Input: chunks created in the previous step
input_file = Path("chunks") / "document_chunks.json"

# Output: chunks with their embeddings
output_folder = Path("embedded_data")
output_folder.mkdir(exist_ok=True)
output_file = output_folder / "embedded_chunks.json"

# 1. Load the chunks
with input_file.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No chunks found in document_chunks.json")

print(f"Total chunks loaded: {len(chunks)}")

# 2. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. Extract the text from every chunk
texts = [chunk["text"] for chunk in chunks]

# 4. Convert each text into a numerical vector
print("Generating embeddings...")
vectors = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 5. Attach each vector to its original chunk
for chunk, vector in zip(chunks, vectors):
    chunk["embedding"] = vector.tolist()

# 6. Save the results
with output_file.open("w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False)

print("\nEmbedding generation completed!")
print(f"Chunks processed: {len(chunks)}")
print(f"Vector dimensions: {len(chunks[0]['embedding'])}")
print(f"Output saved to: {output_file.resolve()}")