import os
import pypdf
import chromadb
from dotenv import load_dotenv
from utils import get_embedding, chunk_text

load_dotenv()

PDF_PATH = "data/General_Psychology.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "psychology"

def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"Readin PDF: {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""

    for i, page in enumerate(reader.pages):        
        text = page.extract_text()
        if text:
            full_text += text + " "
        if (i + 1) % 50 == 0:
            print(f"Extracted text from {i + 1}/{len(reader.pages)} pages...")
    print(f"Done. Total pages: {len(reader.pages)}")
    return full_text

def ingest():
    #1. extract text from PDF
    raw_text = extract_text_from_pdf(PDF_PATH)

    #2. Chunk it
    print("Chunking text...")
    chunks = chunk_text(raw_text)
    print(f"created {len(chunks)} chunks")

    #3. Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    #4. Embed and store in batches
    print("Embedding and storing chunks...")
    BATCH_SIZE=50

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"chunk_{j}" for j in range(i, i+len(batch))]
        embeddings = get_embedding(batch)

        collection.add(
            ids = ids,
            documents=batch,
            embeddings=embeddings
        )

        print(f"stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")

    print(f"\nDone! {len(chunks)} chunks ingested into ChromaDB collection '{COLLECTION_NAME}'. ")

if __name__ == "__main__":    
    ingest()