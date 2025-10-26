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
def chunk_policy_text(full_text,chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP):
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
                "text": f"{header}\n{sub}",
                "section": header
            })

    return final_chunks

# Loading all policy text files
texts, metadata = [], []

for file in POLICY_OUTPUT_DIR.glob("*.txt"):
    content = file.read_text(encoding="utf-8").strip()
    if not content:
        print(f"Skipping empty file: {file.name}")
        continue

    policy_id = os.path.basename(file).replace(".txt", "")
    chunks = chunk_policy_text(content)

    for chunk in chunks:
        texts.append(chunk["text"])
        metadata.append({
            "policy_id": policy_id,
            "section": chunk["section"]
        })

print(f"Loaded {len(metadata)} total chunks from {len(list(POLICY_OUTPUT_DIR.glob('*.txt')))} policies")

# Creating embedding model using SapBERT
embeddings = HuggingFaceEmbeddings(model_name=POLICY_EMBED_MODEL)

if texts:
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadata,
        persist_directory=str(PERSIST_DIR)
    )
    vectordb.persist()
    print(f"Embeddings saved to: {str(PERSIST_DIR)}")
else:
    print("No valid text files found.")