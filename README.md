# PolicyMind Application

**A Retrieval-Augmented Generation System for Intelligent Policy Analysis**

> *"Thinks through policies intelligently."*

This project was developed as part of the **University of South Florida’s _Text Analytics_ course**: **Week 8 – Retrieval-Augmented Generation Foundations & Team Assignment**) under the guidance of **Dr. Tim Smith**.

PolicyMind is a RAG-enhanced conversational AI application designed to interpret complex insurance policies used in clinical decision-making and prior authorization processes. By combining hybrid search with generative reasoning, PolicyMind helps healthcare professionals quickly access authoritative policy information with verifiable evidence citations.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [RAG Architecture](#rag-architecture)
- [Security Features](#security-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Hybrid Retrieval System**: Combines semantic search, query expansion, and cross-encoder re-ranking for optimal policy retrieval
- **Medical Policy Database**: Pre-loaded with 55 insurance policies from major payers (Anthem, Aetna, UnitedHealthcare, Molina, CMS)
- **Source Attribution**: Every response includes inline citations [1], [2], [3] linking back to specific policy documents
- **Security Guardrails**: Defense-in-depth approach protecting against prompt injection, jailbreaking, and domain misuse
- **Conversational Interface**: Multi-turn conversations with context preservation and session management
- **Token Management**: Real-time token counting with warnings when approaching context limits
- **Message Regeneration**: Ability to regenerate responses with different models or prompts
- **User Authentication**: Secure session management with user-specific conversation history
- **Policy Summarization**: Automatic generation of structured policy summaries with citations

---

## Demo Video

**Watch the 5-minute demonstration video here https://youtu.be/pB48ra6Twnc**

The demo showcases:
- Application walkthrough and key features
- RAG functionality with source attribution
- Security features (prompt injection defense)
- Context window handling
- Real-world use cases

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** (tested on Python 3.9, 3.10, 3.11)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **OpenAI API Key** (required for LLM functionality)
- **8GB+ RAM** (recommended for embedding models)
- **5GB+ disk space** (for vector database and models)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mudejayaprakash/AlphaNeurons-RAG-Chatbot.git
```

### 2. Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes as it downloads large embedding models.

### 4. Download Required Models

The following models will be automatically downloaded on first run:
- **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) - ~440MB
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) - ~80MB

---
## Project Structure

```
AlphaNeurons-RAG-Chatbot/
├── README.md                    # Setup instructions and overview (This file)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
│
├── app.py                       # Main Streamlit application
├── chat_app.db                  # Streamlit application database
├── data_ingestion.py            # Data processing pipeline
│
├── utils/                       # Core functionality modules
│   ├── config.py                # Configuration and constants
│   ├── rag.py                   # RAG retrieval and generation
│   ├── security.py              # Security and validation
│   └── database.py              # Vector database interactions
│
├── data/                        # Data storage (git-ignored)
│   ├── raw_policy_pdf/          # Original PDF files
│   ├── policy_txt/              # Processed text files
│   └── embeddings/              # ChromaDB vector store
│
├── PolicyMindAPP.pdf            # Written Report

```
---

## Configuration

### 1. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Override default model
OPENAI_MODEL=gpt-4o-mini

# Optional: Adjust rate limits
RATE_LIMIT_TURNS=20
```

### 2. Configure Data Directories

By default, PolicyMind uses the following structure:

```
data/
├── raw_policy_pdf/      # Place your PDF policy files here
├── policy_txt/          # Processed text files (auto-generated)
└── embeddings/          # Vector database (auto-generated)
```

**To customize paths**, edit `utils/config.py`:

```python
BASE_DIR = Path("/your/custom/path/data")
```

### 3. Verify Configuration

Run the configuration test:

```bash
python -c "from utils.config import *; print('Configuration loaded successfully')"
```

---

## Usage

### Step 1: Prepare Your Data

Place PDF policy files in `data/raw_policy_pdf/`:

```bash
# Example: Download sample policies
mkdir -p data/raw_policy_pdf
# Place your .pdf files here
```

### Step 2: Run Data Ingestion

Process PDFs and build the vector database:

```bash
python data_ingestion.py
```

**Expected output:**
```
Processing: policy_name.pdf
Saved: data/policy_txt/policy_name.txt
...
Loaded 3491 total chunks from 55 policies
Creating Chroma vector DB at: data/embeddings
Embeddings successfully saved
```

**⏱️ Time estimate**: ~2-5 minutes for 55 policies

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`

### Step 4: Using PolicyMind

1. **Login/Resgister Account**: Enter username and password
2. **Search Policies**: Use the search box to find relevant policies
3. **Ask Questions**: Type questions about policy coverage, requirements, exclusions
4. **View Citations**: Click references for citations numbers mapping to corresponding pages from source documents
5. **Regenerate**: Use the 🔄 button to try different models or prompts

---

## Data Pipeline

### Overview

PolicyMind uses a multi-stage data pipeline:

```
PDF Files → Text Extraction → Section Detection → Chunking → Embedding → Vector DB
```

### Stage 1: PDF to Text Conversion

```python
# Uses UnstructuredPDFLoader from LangChain
loader = UnstructuredPDFLoader(pdf_path)
text = loader.load()
```

### Stage 2: Section-Aware Chunking

- **Strategy**: Regex-based section detection + token-controlled splitting
- **Chunk Size**: ~800 tokens with 100-token overlap
- **Metadata**: Preserves source, section, and page information

```python
# Detects common policy headers
pattern = r"Medical Necessity Criteria|Coverage Limitations|..."
sections = re.split(pattern, full_text)
```

### Stage 3: Embedding Generation

- **Model**: SapBERT (biomedical domain-adapted)
- **Dimensions**: 768-dimensional vectors
- **Storage**: ChromaDB with local persistence

### Customizing the Pipeline

To process new policies:

1. Add PDFs to `data/raw_policy_pdf/`
2. Run `python data_ingestion.py`
3. Restart the Streamlit app

---

## RAG Architecture

### Retrieval Strategy

PolicyMind uses a **three-stage hybrid retrieval system**:

#### Stage 1: Query Expansion
- **LLM-Powered**: Generates synonyms and abbreviations
- **Example**: "knee replacement" → "total knee arthroplasty, TKA, joint replacement"

#### Stage 2: Semantic Search
- **Embedding Model**: SapBERT for medical terminology
- **Initial Retrieval**: Top k candidates via cosine similarity

#### Stage 3: Cross-Encoder Re-Ranking
- **Model**: `ms-marco-MiniLM-L-6-v2`
- **Final Selection**: Top k most contextually relevant chunks

### Document-Level Aggregation

- **Strategy**: Maximum chunk score per policy
- **Rationale**: Prioritizes policies with at least one highly relevant section

### Context Integration

```python
# Citation mapping
[1] Medical necessity requires documented obstruction...
[2] Contraindications include active infection...

# Structured prompts with inline citations
context = build_citation_map(retrieved_chunks)
```

---

## Security Features

### Defense-in-Depth Architecture

PolicyMind implements multiple security layers:

#### 1. Input Validation
- **Prompt Injection Detection**: Regex patterns for "ignore previous", "act as", etc.
- **Domain Validation**: Rejects medical advice or PHI-related queries
- **Gibberish Filtering**: Detects non-linguistic input

#### 2. Rate Limiting
- **Query Length**: Max 1000 characters
- **Conversation Depth**: Max 20 turns per session
- **Token Budget**: Warnings at 100K tokens

#### 3. Output Validation
- **Citation Enforcement**: Requires inline citations for all facts
- **Content Scanning**: Removes unsafe or directive language
- **Source Verification**: Links citations to actual document chunks

#### 4. Logging & Monitoring
- **Security Events**: Logs all blocked attempts
- **Audit Trail**: Timestamps and reasons for rejections

---

## Troubleshooting

### Common Issues

#### 1. "OPENAI_API_KEY not found"

**Solution:**
```bash
# Verify .env file exists
ls -la .env

# Check contents
cat .env

# Ensure format is correct (no quotes needed)
OPENAI_API_KEY=sk-proj-...
```

#### 2. "No module named 'utils'"

**Solution:**
```bash
# Ensure you're in the project root directory
pwd  # Should end in /AlphaNeurons-RAG-Chatbot

# Verify utils directory exists
ls utils/

# Reinstall in development mode
pip install -e .
```

#### 3. "ChromaDB: No such file or directory"

**Solution:**
```bash
# Run data ingestion to create the database
python data_ingestion.py

# Verify embeddings directory was created
ls data/embeddings/
```

#### 4. "Model download failed"

**Solution:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Reinstall sentence-transformers
pip install --upgrade sentence-transformers

# Retry data ingestion
python data_ingestion.py
```

#### 5. "Streamlit port already in use"

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the existing process
lsof -ti:8501 | xargs kill -9
```

### Performance Issues

**Slow retrieval times?**
- Reduce `MAX_CANDIDATES` in `config.py` (try 50 instead of 100)
- Use a smaller embedding model (though less accurate)

**High memory usage?**
- Close other applications
- Reduce chunk size in `config.py`
- Use batch processing for large datasets

### Getting Help

If you encounter issues not covered here please review the [technical report](PolicyMindAPP.pdf) for detailed architecture

---

##  Acknowledgments

### Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[LangChain](https://www.langchain.com/)** - LLM application framework
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[OpenAI](https://openai.com/)** - GPT-4 models
- **[Sentence Transformers](https://www.sbert.net/)** - Embedding models
- **[SapBERT](https://github.com/cambridgeltl/sapbert)** - Biomedical embeddings

### Data Sources

Medical policy documents sourced from:
- Anthem Blue Cross Blue Shield
- Aetna
- UnitedHealthcare
- Molina Healthcare
- Centers for Medicare & Medicaid Services (CMS)

All policies are publicly available and used in accordance with their respective terms of service.

### Course Information

This project was developed as part of:
- **Course**: ISM 6564 - Text Analytics
- **Institution**: University of South Florida
- **Instructor**: Dr. Tim Smith
- **Semester**: Fall 2025

## Citation

If you use PolicyMind in your research or project, please cite:

```bibtex
@software{policymind2025,
  author = {Alpha Neurons},
  title = {PolicyMind: A Retrieval-Augmented Generation System for Intelligent Policy Analysis},
  year = {2025},
  url = {https://github.com/mudejayaprakash/AlphaNeurons-RAG-Chatbot}
}
```


