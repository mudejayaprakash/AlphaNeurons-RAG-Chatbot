import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Base directory
BASE_DIR = Path("/Users/sanjana/Desktop/General/Text_Analytics/w8/Assignment_raw/w8_Assignment/data")

# Directory structure
POLICY_INPUT_DIR = BASE_DIR / "raw_policy_pdf"
POLICY_OUTPUT_DIR = BASE_DIR / "policy_txt"
PERSIST_DIR = BASE_DIR / "embeddings"

# Embedding models
POLICY_EMBED_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# RAG parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CANDIDATES = 100
RATE_LIMIT_TURNS = 2

# Loading environment variables from .env file
load_dotenv()

# Verifying API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Initializing OpenAI client
openai_client = OpenAI(api_key=api_key)

# Chose the model
open_ai_model = "gpt-4o-mini"

# Initializing CrossEncoder
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
print(f"CrossEncoder loaded: {CROSS_ENCODER_MODEL}")