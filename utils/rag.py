from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
import traceback
import numpy as np
import re
from utils.config import PERSIST_DIR, POLICY_EMBED_MODEL, open_ai_model,cross_encoder,MAX_CANDIDATES, RATE_LIMIT_TURNS

# Import security validation functions
try:
    from utils.security import validate_user_input, validate_output, log_security_event
    SECURITY_ENABLED = True
except ImportError:
    print("Warning: security.py not found. Security checks disabled.")
    SECURITY_ENABLED = False

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

    if isinstance(queries, str):
        queries = [queries]

    for q in queries:
        if not q:
            continue
        hits = db.similarity_search(q, k=k)  # returns List[Document]
        for d in hits:
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
        if isinstance(item, tuple):
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

# Helper function to build citation map
def _build_citation_map(retrieved_chunks):
    """Build citation map from retrieved chunks"""
    citation_map = {}
    context_blocks = []
    
    for idx, c in enumerate(retrieved_chunks, 1):
        if isinstance(c, dict):
            metadata = c.get("metadata", {})
            text = c.get("page_content", "")
        else:
            metadata = getattr(c, "metadata", {})
            text = getattr(c, "page_content", "")

        source = metadata.get("source") or "Unknown_Source"
        page = metadata.get("page") or "?"

        citation_key = f"[{idx}]"
        citation_map[citation_key] = {
            "source": source,
            "page": page,
            "full_citation": f"{source}, p.{page}"
        }
        context_blocks.append(f"{citation_key} {text}")

    context_text = "\n\n".join(context_blocks)
    return context_text, citation_map

# Helper function to group citations by document
def group_citations_by_document(citation_map):
    """
    Group citations by document name
    Returns: dict with document names as keys and list of (citation_key, page) tuples as values
    Example: {
        "policy_abc.pdf": [("[1]", "5"), ("[3]", "12")],
        "policy_xyz.pdf": [("[2]", "8")]
    }
    """
    grouped = {}
    for cite_key, cite_info in citation_map.items():
        doc_name = cite_info["source"]
        page = cite_info["page"]
        
        if doc_name not in grouped:
            grouped[doc_name] = []
        grouped[doc_name].append((cite_key, page))
    
    return grouped

# Summarize the main coverage, exclusions, and medical necessity criteria from the given policy chunks.
def summarize_policy_chunks(retrieved_chunks, llm_client=None, llm_model: str =open_ai_model):
    if not llm_client:
        return "LLM client unavailable. Please try again later."
    if not retrieved_chunks:
        return "No policy content available for summarization."

    # Build context with citations
    context_text, citation_map = _build_citation_map(retrieved_chunks)

    system_msg=(
        "You are a healthcare policy summarization assistant. "
        "Summarize the policy content concisely using bullet points and section headers "
        "like 'Coverage Criteria', 'Medical Necessity Conditions', and 'Exclusions'. "
        "Use inline numeric citations (e.g., [1], [2]) to indicate the source "
        "of each fact or rule. Do not include extra commentary or interpretation."
    )

    user_msg = f"Policy Context:\n{context_text}\n\nSummarize the key criteria clearly."

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
        summary_text = response.choices[0].message.content.strip()

        return {
            "summary": summary_text,
            "references": citation_map
        }

    except Exception as e:
        print("Summarization failed:", e)
        traceback.print_exc()
        return {
            "summary": "Unable to summarize the policy at this time. Please try again later.",
            "references":{}
        }
    
# Multi-turn conversational Q&A - returns answer and references separately
def conversational_policy_qa(retrieved_chunks, user_question: str, llm_client=None, llm_model: str =open_ai_model, 
                             conversation_history: list = None,custom_prompt: str = None):
    if not llm_client:
        return "LLM client unavailable.", conversation_history

    if conversation_history is None:
        conversation_history = []

    # ============================================================
    # SECURITY VALIDATION - Validate user input before processing
    # ============================================================
    if SECURITY_ENABLED:
        is_valid, filtered_question, warning_msg = validate_user_input(
            user_question,
            domain="medical_policy",
            max_length=1000,
            check_injection=True,
            check_domain=True
        )
        
        if not is_valid:
            # Log security event
            log_security_event(
                event_type="INPUT_BLOCKED",
                user_input=user_question,
                reason=warning_msg,
                severity="WARNING"
            )
            # Return warning message without processing
            return warning_msg, conversation_history, False, {}
        
        # Use filtered question (with sensitive content redacted)
        user_question = filtered_question

    # Build context with citations
    context_text, citation_map = _build_citation_map(retrieved_chunks)

    # Use the sidebar-defined custom prompt if provided
    base_prompt = custom_prompt or (
        "You are a healthcare policy assistant. "
        "Use the provided policy context to answer questions accurately. "
        "If the answer is not present, say 'Not found in policy context.' "
        "Maintain continuity from previous conversation turns."
        "Use inline citations (e.g., [1], [2]) when referencing specific information."
    )

    system_msg = {
        "role": "system",
        "content": f"{base_prompt}\n\nPolicy Context:\n{context_text}"
    }

    # Adds user's question to history
    conversation_history.append({"role": "user", "content": user_question})

    # Keep only last 10 exchanges (user+assistant pairs)
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    # Check rate limit
    if len(conversation_history) >= RATE_LIMIT_TURNS * 2:
        # Do NOT add this message as assistant content
        return "RATE_LIMIT_REACHED", conversation_history, True, {}

    messages = [system_msg] + conversation_history
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()

        # ============================================================
        # SECURITY VALIDATION - Validate LLM output before returning
        # ============================================================
        if SECURITY_ENABLED:
            is_safe, filtered_answer, reason = validate_output(
                answer,
                domain="medical_policy"
            )
            
            if not is_safe:
                # Log security event
                log_security_event(
                    event_type="OUTPUT_BLOCKED",
                    user_input=answer,
                    reason=reason,
                    severity="WARNING"
                )
                answer = (
                    "⚠️ I apologize, but I cannot provide that response. "
                    "Please rephrase your question to focus on policy information."
                )
            else:
                # Use filtered answer
                answer = filtered_answer

        # Save assistant reply to history (without references appended)
        conversation_history.append({"role": "assistant", "content": answer})
        
        return answer, conversation_history, False, citation_map

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
        return answer, conversation_history, rate_limit, {}

# Filters ranked results to return only chunks belonging to one policy - useful for Streamlit UI
def get_chunks_for_policy(ranked_results, policy_id: str):
    return [c for c, _ in ranked_results if c.metadata.get("policy_id") == policy_id]
