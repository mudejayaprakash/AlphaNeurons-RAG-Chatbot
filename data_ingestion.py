from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import warnings, os, sys, re, glob
from utils.config import POLICY_INPUT_DIR, POLICY_OUTPUT_DIR, PERSIST_DIR, POLICY_EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# Suppress all warnings
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, "w")

# Create directories if they don't exist
for directory in [POLICY_INPUT_DIR, POLICY_OUTPUT_DIR, PERSIST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Converting downloaded policy PDF files to .txt format
for pdf_file in POLICY_INPUT_DIR.glob("*.pdf"):
    print(f"Processing: {pdf_file.name}")
    try:
        loader = UnstructuredPDFLoader(str(pdf_file))
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
        output_path = POLICY_OUTPUT_DIR / f"{pdf_file.stem}.txt"
        output_path.write_text(text.strip(), encoding="utf-8")
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {pdf_file.name}: {e}")

print("All PDFs processed and saved to:", POLICY_OUTPUT_DIR)

# Creating Chunks using regex-based section detection with token controlled chunking
def chunk_policy_text(full_text,file_path=None,chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP):
    """
    Combines regex-based section detection with
    token-controlled chunking (≈800 tokens, 100 overlap).
    Returns a list of chunk dicts ready for embedding.
    """
    # Regex pattern for common policy headers
    pattern = re.compile(
        r"(?m)(?:^|\n)([A-Za-z][A-Za-z\s]{3,}?"
        r"(?:Medical Necessity Criteria|Coverage Limitations|Documentation Requirements|"
        r"Contraindications|Summary|Policy History|References|Coding Information|Scope)[:]?)",
        flags=re.IGNORECASE
    )

    # Splitting document into section header + content pairs
    sections = re.split(pattern, full_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = []

    source_name = os.path.basename(file_path) if file_path else "Unknown_Source"

    # Iterating through sections (every even index = header, odd = content)
    for i in range(1, len(sections), 2):
        header = sections[i].strip().title()  # normalize header text
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if not content:
            continue

        # Applying token-controlled chunking within each section
        sub_docs = splitter.split_text(content)
        for j, sub in enumerate(sub_docs):
            final_chunks.append({
                "id": f"{header}_{j}",
                "page_content": f"{header}\n{sub}",
                "metadata": {
                    "source": source_name,     
                    "page": j + 1,             
                    "section": header       
                }
            })

    return final_chunks

# Database Creation- Ingestion Pipeline
def build_vector_database():
    """
    Reads all .txt policy files, chunks them with metadata, embeds into Chroma DB, and persists locally.
    """
    all_docs = []
    embedder = HuggingFaceEmbeddings(model_name=POLICY_EMBED_MODEL)

    for file in POLICY_OUTPUT_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"Skipping empty file: {file.name}")
            continue

        policy_id = os.path.basename(file).replace(".txt", "")
        chunks = chunk_policy_text(content, file_path=file.name)

        for idx, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            metadata = {
                "policy_id": policy_id,
                "source": f"{policy_id}.pdf",
                "page": idx + 1,
                "section": meta.get("section", "Unknown_Section")
            }

            all_docs.append(Document(
                page_content=chunk.get("page_content") or chunk.get("text", ""),
                metadata=metadata
            ))

    print(f"Loaded {len(all_docs)} total chunks from {len(list(POLICY_OUTPUT_DIR.glob('*.txt')))} policies")

    if all_docs:
        print(f"Creating Chroma vector DB at: {PERSIST_DIR}")
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=embedder,
            persist_directory=str(PERSIST_DIR)
        )
        vectordb.persist()
        print(f"Embeddings successfully saved to: {PERSIST_DIR}")
    else:
        print("No valid text files found for ingestion.")


# Run the ingestion when file executed
if __name__ == "__main__":
    build_vector_database()

#Debug function - use as an alternative to above function while debugging    
# if __name__ == "__main__":
#     # Temporarily enable error messages
#     sys.stderr = sys.__stderr__
#     warnings.filterwarnings("default")
    
#     print("🚀 Starting policy ingestion...")
    
#     # Debug info
#     print(f"📁 Input dir: {POLICY_INPUT_DIR}")
#     print(f"📁 Output dir: {POLICY_OUTPUT_DIR}")
#     print(f"📁 Persist dir: {PERSIST_DIR}")
    
#     # Check for PDFs
#     pdf_files = list(POLICY_INPUT_DIR.glob("*.pdf"))
#     print(f"📄 Found {len(pdf_files)} PDF files")
    
#     # Check for TXT files
#     txt_files = list(POLICY_OUTPUT_DIR.glob("*.txt"))
#     print(f"📝 Found {len(txt_files)} TXT files")
    
#     if len(txt_files) == 0:
#         print("⚠️  No text files found! Uncomment PDF processing code first.")
#     else:
#         build_vector_database()
#         print("✅ Done building vector database!")