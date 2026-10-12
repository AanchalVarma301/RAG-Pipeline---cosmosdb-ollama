
from pathlib import Path
from pypdf import PdfReader

# Folder containing the original PDF documents
data_folder = Path("data")

# Folder where extracted text will be saved
output_folder = Path("extracted_text")
output_folder.mkdir(exist_ok=True)

# Process both PDFs
pdf_files = [
    data_folder / "human-rights-statement.pdf",
    data_folder / "accenture-cobe-brochure-english.pdf",
]

for pdf_path in pdf_files:
    print(f"\nProcessing: {pdf_path.name}")

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        continue

    reader = PdfReader(str(pdf_path))
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        extracted_pages.append(
            f"\n--- Page {page_number} ---\n{page_text}"
        )

    # Combine page text and save it
    full_text = "\n".join(extracted_pages)

    output_path = output_folder / f"{pdf_path.stem}.txt"
    output_path.write_text(full_text, encoding="utf-8")

    print(f"Pages found: {len(reader.pages)}")
    print(f"Characters extracted: {len(full_text)}")
    print(f"Saved extracted text to: {output_path}")

    if not full_text.strip():
        print(
            "WARNING: No text was extracted. "
            "This PDF may need OCR if it contains scanned pages."
        )

print("\nPDF extraction process completed.")