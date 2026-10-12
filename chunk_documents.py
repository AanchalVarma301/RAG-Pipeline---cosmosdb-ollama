from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Folder containing the extracted PDF text
input_folder = Path("extracted_text")

# 2. Folder where chunks will be saved
output_folder = Path("chunks")
output_folder.mkdir(exist_ok=True)

# 3. Configure the chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 4. Process both extracted documents
text_files = list(input_folder.glob("*.txt"))

if not text_files:
    raise FileNotFoundError(
        "No .txt files found in the extracted_text folder."
    )

all_chunks = []

for text_file in text_files:
    print(f"\nProcessing: {text_file.name}")

    # Read extracted text
    text = text_file.read_text(encoding="utf-8")

    if not text.strip():
        print("Warning: File is empty. Skipping.")
        continue

    # Split the document into smaller chunks
    document_chunks = text_splitter.split_text(text)

    for chunk_number, chunk_text in enumerate(document_chunks, start=1):
        all_chunks.append({
            "id": f"{text_file.stem}-chunk-{chunk_number}",
            "document_name": text_file.name,
            "chunk_number": chunk_number,
            "text": chunk_text
        })

    print(f"Chunks created: {len(document_chunks)}")

# 5. Save all chunks in JSON format
output_file = output_folder / "document_chunks.json"

output_file.write_text(
    json.dumps(all_chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"\nTotal chunks created: {len(all_chunks)}")
print(f"Saved to: {output_file.resolve()}")