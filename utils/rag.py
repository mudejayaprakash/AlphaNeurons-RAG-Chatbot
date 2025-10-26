from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
import traceback
import numpy as np
import re
from utils.config import PERSIST_DIR, POLICY_EMBED_MODEL, open_ai_model,cross_encoder,MAX_CANDIDATES, RATE_LIMIT_TURNS

# Loading the existing Chroma vector database with the same embedding model used during ingestion.
def load_vector_db(persist_dir: str = PERSIST_DIR):
    try:
        if not isinstance(persist_dir, str):
            persist_dir = str(persist_dir)
        embeddings = HuggingFaceEmbeddings(model_name=POLICY_EMBED_MODEL)
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("Loaded Chroma collection successfully!")
        print("Total documents embedded:", db._collection.count())
        return db

    except Exception as e:
        print(f"Failed to load vector DB: {e}")
        return None

# Expands a medical query using an LLM if available.
# Falls back to the direct query only if expansion fails or no client is provided.
def expand_query(user_query: str, llm_client=None, llm_model: str =open_ai_model,provider: str = "openai") -> list[str]:
    base = user_query.strip()
    variants = [base]

    if llm_client:
        try:
            # Defining base messages
            system_msg = (
                "You are a medical terminology assistant. "
                "When given a medical concept, return a concise comma-separated list of synonyms, "
                "abbreviations, and related expressions. Do not explain anything."
            )
            user_msg = f"Concept: '{base}'"

            # OpenAI call
            if provider.lower() == "openai":
                resp = llm_client.chat.completions.create(
                    model=llm_model,  # or your preferred model
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0,
                    max_tokens=150
                )
                text = resp.choices[0].message.content.strip()

            # Parse response
            if not text:
                raise ValueError("Empty model response")
            if text.startswith("["):
                try:
                    terms = json.loads(text)
                except Exception:
                    terms = [t.strip(" '\"") for t in re.split(r"[,\n]", text) if t.strip()]
            elif re.search(r"^\d+\.", text):
                terms = [re.sub(r"^\d+\.\s*", "", line).strip() for line in text.splitlines() if line.strip()]
            else:
                terms = [t.strip(" '\"") for t in text.split(",") if t.strip()]
            variants.extend(terms)
            print(f"Query expanded via {provider}: {variants}")

        except Exception as e:
            print(f"LLM expansion failed ({provider}). Using direct query only.\nError: {e}")

    else:
        print("No LLM client provided — using direct query only.")

    # Deduplicating while preserving order
    seen, deduped = set(), []
    for v in variants:
        vl = v.lower()
        if vl not in seen:
            seen.add(vl)
            deduped.append(v)

    return deduped

# Multi-query semantic search against the vector DB. Returns a flat list of LangChain Document objects (with metadata preserved)
def semantic_search(db, queries, k=20):
    all_hits = []
    seen = set()  # de-duplicate by (source, page) if present

    # Normalize queries (list[str])
    if isinstance(queries, str):
        queries = [queries]

    for q in queries:
        if not q:
            continue
        hits = db.similarity_search(q, k=k)  # returns List[Document]
        for d in hits:
            # Build a dedup key based on metadata if possible
            meta = getattr(d, "metadata", {}) or {}
            key = (meta.get("policy_id") or meta.get("source") or meta.get("doc_id") or meta.get("file_name") or "UNK",
                   meta.get("page") or meta.get("section") or "")
            if key in seen:
                continue
            seen.add(key)
            all_hits.append(d)
    return all_hits

# Using Cross-Encoder to rerank the retrieved chunks
def re_rank_crossencoder(query: str, candidates):
    """Score (query, passage) pairs and return top_k Documents."""
    pairs = [(query, c.page_content) for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked

#   Aggregating chunk-level scores to document-level rankings.
def aggregate_to_documents(ranked_chunks, top_k_docs: int = 5):
    doc_scores = defaultdict(list)

    for item in ranked_chunks:
        # handle both tuple (doc, score) and dict {"score":..., "metadata":...}
        if isinstance(item, tuple):
            # determine which element is the Document
            doc, score = (item if hasattr(item[0], "metadata") else (item[1], item[0]))
            meta = getattr(doc, "metadata", {}) or {}
            val = float(score)
        else:
            meta = item.get("metadata", {}) or {}
            val = float(item.get("score", 0.0))

        policy_id = (
            meta.get("policy_id")
            or meta.get("source")
            or meta.get("doc_id")
            or meta.get("file_name")
            or "Unknown Document"
        )
        doc_scores[policy_id].append(val)

    # Max score per document
    doc_final = {doc: np.max(scores) for doc, scores in doc_scores.items()}

    # Sort descending by document score
    ranked_docs = sorted(doc_final.items(), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked_docs[:top_k_docs]]

# Retrieving the top contexts
def retrieve_top_contexts(user_query: str, db: Chroma, llm_client=None, provider = "openai", top_k_final: int = 5, max_c=MAX_CANDIDATES):
    """
    1) Expand query variants
    2) Semantic search (broad top-20)
    3) Re-rank chunks
    4) Aggregate chunk-level scores to document-level ranking
    Returns ranked results, variant list and top docs for display/logging.
    """
    # Expanding query variants
    variants = expand_query(user_query, llm_client, provider=provider)

    if not variants:
        print("No valid query variants found. Exiting retrieval.")
        return {"variants": [], "ranked_results": [], "top_docs": []}
    print(f"Expanded queries: {variants}")
    
    # Semantic search
    candidates = semantic_search(db, variants, k=20)
    if not candidates:
        print("No results found in vector database.")
        return {"variants": variants, "ranked_results": [], "top_docs": []}
    print(f"Retrieved {len(candidates)} candidates before re-ranking.")
    
    # Cross-encoder re-ranking of selected number of candidates - expect a list of (doc, score). Provide fallback if unavailable.
    ranked_pairs = []
    candidates = candidates[:max_c]
    try:
        ranked_pairs = re_rank_crossencoder(user_query, candidates)
    except Exception:
        # Fallback: approximate by retrieval order (assign descending pseudo-scores)
        pseudo = float(len(candidates))
        ranked_pairs = [(d, pseudo - i) for i, d in enumerate(candidates)]

    # Normalize into structured chunk dicts
    ranked_chunks = []
    for doc, score in ranked_pairs:
        ranked_chunks.append({
            "page_content": getattr(doc, "page_content", str(doc)),
            "metadata": getattr(doc, "metadata", {}) or {},
            "score": float(score),
        })

    # Aggregate chunk scores to document-level ranking
    top_docs = aggregate_to_documents(ranked_pairs, top_k_docs=top_k_final)

    print(f"Top-{top_k_final} policy documents identified: {top_docs}")

    return {"variants": variants, "ranked_results": ranked_chunks, "top_docs": top_docs}

# Summarize the main coverage, exclusions, and medical necessity criteria from the given policy chunks.
def summarize_policy_chunks(retrieved_chunks, llm_client=None, llm_model: str =open_ai_model) -> str:
    if not llm_client:
        return "LLM client unavailable. Please try again later."
    if not retrieved_chunks:
        return "No policy content available for summarization."

    # Combine all retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    system_msg = (
        "You are a healthcare policy summarization assistant. "
        "Summarize the policy using these section labels like follows if they are relevant: "
        " 'Coverage Criteria', 'Medical Necessity Conditions', and 'Exclusions'. "
        "Return concise bullet points only, with no extra commentary."
    )

    user_msg = f"Policy Context:\n{context}\n\nSummarize the key criteria clearly."

    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Summarization failed:", e)
        return "Unable to summarize the policy at this time. Please try again later."
    
#  Multi-turn conversational Q&A about a selected policy. Keeps previous questions/answers in memory (conversation_history) but attaches the policy context only once.
def conversational_policy_qa(retrieved_chunks, user_question: str, llm_client=None, llm_model: str =open_ai_model, conversation_history: list = None,custom_prompt: str = None):
    if not llm_client:
        return "LLM client unavailable.", conversation_history

    if conversation_history is None:
        conversation_history = []

    # Combine retrieved context (same policy for all turns)
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])

    # Use the sidebar-defined custom prompt if provided
    base_prompt = custom_prompt or (
        "You are a healthcare policy assistant. "
        "Use the provided policy context to answer questions accurately. "
        "If the answer is not present, say 'Not found in policy context.' "
        "Maintain continuity from previous conversation turns."
    )

    system_msg = {
        "role": "system",
        "content": f"{base_prompt}\n\nPolicy Context:\n{context}"
    }

    # Adds user's question to history- not context.
    conversation_history.append({"role": "user", "content": user_question})
    # Keep only last 10 exchanges (user+assistant pairs)
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    # Simulate or enforce rate-limit based on config
    if len(conversation_history) >= RATE_LIMIT_TURNS * 2:
        # Do NOT add this message as assistant content
        return "RATE_LIMIT_REACHED", conversation_history, True

    messages = [system_msg] + conversation_history
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()

        # Save assistant reply to history
        conversation_history.append({"role": "assistant", "content": answer})
        return answer, conversation_history, False

    except Exception as e:
        err_msg = str(e).lower()
        print("Q&A failed:", e)
        traceback.print_exc()

        if "rate limit" in err_msg or "429" in err_msg:
            answer = "**Rate limit reached. Please wait a moment.**"
            rate_limit = True
        elif "context length" in err_msg or "token" in err_msg:
            answer = "**Conversation too long. Please start a new chat to continue.**"
            rate_limit = True
        else:
            answer = f"Unexpected error: {e}"
            rate_limit = False

        # Return the error message gracefully
        conversation_history.append({"role": "assistant", "content": answer})
        return answer, conversation_history, rate_limit

# Filters ranked results to return only chunks belonging to one policy - useful for Streamlit UI
def get_chunks_for_policy(ranked_results, policy_id: str):
    return [c for c, _ in ranked_results if c.metadata.get("policy_id") == policy_id]
