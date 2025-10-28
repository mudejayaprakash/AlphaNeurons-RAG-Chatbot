# app_fixed.py
# UM Policy RAG Assistant — FIXED VERSION

import os
import time
import uuid
import sqlite3
import traceback
import json
from collections import defaultdict
from datetime import datetime

import streamlit as st

# ---- Import your shared OpenAI client and RAG helpers ----
from utils.config import openai_client as oai_client
from utils.rag import (
    load_vector_db, 
    expand_query,            
    semantic_search,        
    retrieve_top_contexts,   
    summarize_policy_chunks,
    conversational_policy_qa,
    group_citations_by_document
)

# Import security functions
try:
    from utils.security import validate_user_input, check_conversation_depth, log_security_event
    SECURITY_ENABLED = True
except ImportError:
    print("Warning: Security module not found. Running without security checks.")
    SECURITY_ENABLED = False

def reset_non_auth_state():
    """Clear everything in session except the auth object."""
    for k in list(st.session_state.keys()):
        if k not in ("auth", "db"):
            del st.session_state[k]

def safe_bool(x):
    return bool(x) if x is not None else False

# =========================
# Session / Auth Utilities
# =========================
DB_PATH = "chat_app.db"

def ensure_session_state():
    ss = st.session_state
    if "auth" not in ss:
        ss.auth = {"logged_in": False, "user_id": None, "preferred_name": None}
    if "db" not in ss:
        try:
            ss.db = load_vector_db()
        except Exception as e:
            ss.db = None
            ss.db_load_error = str(e)
    if "model_sel" not in ss:
        ss.model_sel = "gpt-4o"
    if "conversations" not in ss:
        ss.conversations = {}
    if "current_session_id" not in ss:
        ss.current_session_id = None
    if "last_results" not in ss:
        ss.last_results = None
    if "last_chunks" not in ss:
        ss.last_chunks = []
    if "top_docs" not in ss:
        ss.top_docs = []
    if "last_expansions" not in ss:
        ss.last_expansions = []
    if "selected_policy" not in ss:
        ss.selected_policy = None
    if "selected_policy_chunks" not in ss:
        ss.selected_policy_chunks = []
    if "policy_summary" not in ss:
        ss.policy_summary = None
    if "want_regen" not in ss:         
        ss.want_regen = False
    if "view_mode" not in ss:
        ss.view_mode = "search"  # "search" or "conversation"
    if "viewing_policy" not in ss:
        ss.viewing_policy = None
    if "is_read_only" not in ss:
        ss.is_read_only = False  # 

def user_key(key: str) -> str:
    """Prefix session keys by username to isolate user data."""
    uid = st.session_state.auth.get("user_id") or "guest"
    return f"{uid}_{key}"


def init_user_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        preferred_name TEXT
    )
    """)
    
    # UPDATED: Support multiple conversations per policy
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        policy_name TEXT,
        session_id TEXT UNIQUE,
        summary TEXT,
        summary_references TEXT,
        chunks TEXT,
        chat_references TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)
    
    # Table for storing chat messages
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    )
    """)
    
    # ---- Schema migrations (for older DBs) ----
    try:
        c.execute("SELECT preferred_name FROM users LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE users ADD COLUMN preferred_name TEXT")
    
    # Add session_id column if it doesn't exist
    try:
        c.execute("SELECT session_id FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
        # Generate session IDs for existing rows
        c.execute("UPDATE conversations SET session_id = hex(randomblob(16)) WHERE session_id IS NULL")
    
    # Add chat_references column if it doesn't exist
    try:
        c.execute("SELECT chat_references FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE conversations ADD COLUMN chat_references TEXT")
        print("✅ Added chat_references column to conversations table")

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_conversations_username 
    ON conversations(username)
    """)
    
    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_conversations_session 
    ON conversations(session_id)
    """)
    
    conn.commit()
    conn.close()

def save_conversation(username, policy_name, summary, summary_refs, chunks, history, chat_refs=None, session_id=None):
    """Save or update a conversation in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        chunks_json = json.dumps(chunks)
        summary_refs_json = json.dumps(summary_refs) if summary_refs else "{}"
        chat_refs_json = json.dumps(chat_refs) if chat_refs else "{}"
        
        # Check if this session already exists
        c.execute("""
            SELECT id FROM conversations 
            WHERE session_id=?
        """, (session_id,))
        result = c.fetchone()
        
        if result:
            # Update existing conversation
            conv_id = result[0]
            c.execute("""
                UPDATE conversations 
                SET summary=?, summary_references=?, chunks=?, chat_references=?, created_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (summary, summary_refs_json, chunks_json, chat_refs_json, conv_id))
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        else:
            # Create new conversation
            c.execute("""
                INSERT INTO conversations (username, policy_name, session_id, summary, summary_references, chunks, chat_references)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, policy_name, session_id, summary, summary_refs_json, chunks_json, chat_refs_json))
            conv_id = c.lastrowid
        
        # Save messages
        if history:
            for msg in history:
                c.execute("""
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (?, ?, ?)
                """, (conv_id, msg.get("role"), msg.get("content")))
        
        conn.commit()
        print(f"✅ Saved conversation (session: {session_id[:8]}...) for policy '{policy_name}'")
        return session_id
    
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        traceback.print_exc()
        conn.rollback()
        return None
    finally:
        conn.close()

def load_cached_summary(username, policy_name):
    """Load ONLY the cached summary for a policy, not the conversations"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Get the most recent summary for this policy
        c.execute("""
            SELECT summary, summary_references, chunks
            FROM conversations
            WHERE username=? AND policy_name=?
            ORDER BY created_at DESC
            LIMIT 1
        """, (username, policy_name))
        
        result = c.fetchone()
        if result:
            summary, summary_refs_json, chunks_json = result
            summary_refs = json.loads(summary_refs_json) if summary_refs_json else {}
            chunks = json.loads(chunks_json) if chunks_json else []
            return summary, summary_refs, chunks
        
        return None, None, None
    
    except Exception as e:
        print(f"Error loading cached summary: {e}")
        return None, None, None
    
    finally:
        conn.close()

def load_user_conversations(username):
    """Load all saved conversations for sidebar display - ONLY those with messages"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    conversations = []
    
    try:
        c.execute("""
            SELECT id, policy_name, session_id, summary, summary_references, chunks, chat_references, created_at
            FROM conversations
            WHERE username=?
            ORDER BY created_at DESC
        """, (username,))
        
        for row in c.fetchall():
            conv_id, policy_name, session_id, summary, summary_refs_json, chunks_json, chat_refs_json, created_at = row
            
            # Get messages for this conversation
            c.execute("""
                SELECT role, content FROM messages
                WHERE conversation_id=?
                ORDER BY timestamp ASC
            """, (conv_id,))
            
            messages = [{"role": role, "content": content} for role, content in c.fetchall()]
            
            # Only include conversations with messages
            if not messages:
                continue
            
            try:
                summary_refs = json.loads(summary_refs_json) if summary_refs_json else {}
            except:
                summary_refs = {}
            
            try:
                chunks = json.loads(chunks_json) if chunks_json else []
            except:
                chunks = []
            
            try:
                chat_refs = json.loads(chat_refs_json) if chat_refs_json else {}
            except:
                chat_refs = {}
            
            conversations.append({
                "session_id": session_id,
                "policy_name": policy_name,
                "summary": summary or "",
                "summary_refs": summary_refs,
                "chunks": chunks,
                "history": messages,
                "message_references": chat_refs,
                "created_at": created_at
            })
    
    except Exception as e:
        print(f"Error loading conversations: {e}")
        traceback.print_exc()
    
    finally:
        conn.close()
    
    return conversations

def validate_login(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, preferred_name FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    if result:
        return True, result[0], result[1] or result[0]
    return False, None, None

def register_user(username, password, preferred_name=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, preferred_name) VALUES (?, ?, ?)", 
                 (username, password, preferred_name or username))
        conn.commit()
        conn.close()
        return True, f"User '{username}' registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Username '{username}' already exists. Please choose a different one."

def compute_doc_stats(chunks):
    doc_set = set()
    for c in chunks:
        if not isinstance(c, dict):
            continue
        meta = c.get("metadata", {}) or {}
        doc_id = (
            meta.get("policy_id")
            or meta.get("source")
            or meta.get("doc_id")
            or meta.get("file_name")
            or "Unknown"
        )
        doc_set.add(doc_id)
    doc_ranked = sorted(doc_set)
    return len(chunks), len(doc_set), doc_ranked


# ===============================================
# Main App Logic
# ===============================================
st.set_page_config(page_title="PolicyMind", page_icon="📘", layout="wide")

init_user_table()
ensure_session_state()

# Session persistence - Do NOT auto-load conversations, only set auth
if "user" in st.query_params and not st.session_state.auth["logged_in"]:
    username = st.query_params["user"]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, preferred_name FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        st.session_state.auth = {"logged_in": True, "user_id": result[0], "preferred_name": result[1] or result[0]}
        # Conversations are NOT auto-loaded here
        # They are only loaded when user explicitly clicks on them from sidebar

uid = st.session_state.auth.get("user_id") or "guest"

# ===============================================
# Sidebar
# ===============================================
with st.sidebar:
    if st.session_state.auth["logged_in"]:
        preferred_name = st.session_state.auth.get("preferred_name") or st.session_state.auth["user_id"]
        st.markdown(f"### Welcome, {preferred_name}!")
        
        st.divider()
        st.subheader("⚙️ Settings")
        st.session_state.model_sel = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4o-mini", "o1-preview"],
            index=0
        )
        
        default_prompt = (
            "You are a helpful assistant specializing in medical policy and utilization management. "
            "Answer concisely, accurately, and based on the provided policy context when available."
        )

        st.session_state.system_prompt = st.text_area(
            "System Prompt (for Q&A only)",
            value=st.session_state.get("system_prompt", default_prompt),
            help="Customize the assistant's behavior for question-answering.",
            height=200
        )

        st.divider()
        st.subheader("💬 Previous Conversations")
        
        # Load all conversations from database
        saved_convs = load_user_conversations(st.session_state.auth["user_id"])
        
        if saved_convs:
            for conv in saved_convs:
                # Create display label
                policy_display = conv["policy_name"][:35]
                msg_count = len(conv["history"]) // 2
                label = f"{policy_display}..."
                if msg_count > 0:
                    label += f" ({msg_count} msgs)"
                
                # Get first user message as preview
                preview = ""
                for msg in conv["history"]:
                    if msg["role"] == "user":
                        preview = msg["content"][:50]
                        if len(msg["content"]) > 50:
                            preview += "..."
                        break
                
                # Button to view this conversation
                button_key = f"sidebar_conv_{conv['session_id']}"
                if st.button(label, key=button_key, use_container_width=True, help=preview):
                    # Switch to READ-ONLY conversation view
                    st.session_state.view_mode = "conversation"
                    st.session_state.viewing_policy = conv["policy_name"]
                    st.session_state.viewing_session_id = conv["session_id"]
                    st.session_state.is_read_only = True  # Mark as read-only
                    
                    # Load this specific conversation into session state
                    st.session_state.viewing_conversation = conv
                    st.rerun()
        else:
            st.info("No saved conversations yet. Start chatting with a policy!")
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            username = st.session_state.auth["user_id"]
            
            # Save any active conversations before logout
            if st.session_state.get(user_key("selected_policy")):
                policy_name = st.session_state.get(user_key("selected_policy"))
                chat_key = user_key(f"chat_{policy_name}")
                
                if chat_key in st.session_state and st.session_state[chat_key].get("history"):
                    summary = st.session_state.get(user_key(f"summary_{policy_name}"), "")
                    summary_refs = st.session_state.get(user_key(f"summary_refs_{policy_name}"), {})
                    chunks = st.session_state.get(user_key(f"chunks_{policy_name}"), [])
                    history = st.session_state[chat_key].get("history", [])
                    chat_refs = st.session_state[chat_key].get("message_references", {})
                    session_id = st.session_state[chat_key].get("session_id")
                    
                    save_conversation(username, policy_name, summary, summary_refs, chunks, history, chat_refs, session_id)

            reset_non_auth_state()
            st.session_state.auth = {"logged_in": False, "user_id": None}
            st.query_params.clear()
            
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()


st.markdown(
    """
    <div style="
        background-color: #e0f7f9;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 1rem;
    ">
        <h1 style="
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            color: #319795;
            font-weight: 700;
            font-size: 2.3rem;
            margin-bottom: 0.3rem;
        ">
            PolicyMind
        </h1>
        <p style="
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            color: #4a5568;
            font-size: 1.1rem;
            margin-top: 0;
        ">
            <em>Thinks through policies intelligently.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---- Auth (Login / Register) ----
if not st.session_state.auth["logged_in"]:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])
    with tab1:
        st.subheader("Login")
        uid_input = st.text_input("Username", key="login_uid")
        pwd = st.text_input("Password", key="login_pwd", type="password")
        if st.button("Login"):
            success, user_id, preferred_name = validate_login(uid_input, pwd)
            if success:
                reset_non_auth_state()
                st.session_state.auth = {"logged_in": True, "user_id": uid_input, "preferred_name": preferred_name}
                st.query_params["user"] = uid_input

                st.success(f"Welcome back, {preferred_name}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.subheader("Register New User")
        new_uid = st.text_input("Choose a Username", key="register_uid")
        new_pwd = st.text_input("Choose a Password", key="register_pwd", type="password")
        new_preferred_name = st.text_input("Preferred Name", key="register_preferred_name")
        if st.button("Register"):
            if new_uid and new_pwd:
                success, msg = register_user(new_uid, new_pwd, new_preferred_name or new_uid)
                if success:
                    st.success(msg)
                    time.sleep(1.5)
                else:
                    st.error(msg)
            else:
                st.error("Please enter both a username and password.")
    
    st.stop()

# ===============================================
# Main App (Post-Login)
# ===============================================

if st.session_state.db is None:
    st.error(f"❌ Vector database not loaded. Error: {st.session_state.get('db_load_error', 'Unknown')}")
    st.stop()


# ===============================================
# Render Functions
# ===============================================

def render_search_view():
    """Render the search and policy summary view"""
    st.markdown(
        """
        <div style="
            background-color: #e6f0fa;
            border-left: 4px solid #2b6cb0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
        ">
            <h4 style="
                margin: 0;
                color: #1a202c;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 1.3rem;
            ">
                Policy Search
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    def handle_search_submit():
        query = st.session_state.get(user_key("user_query"), "").strip()
        if query:
            st.session_state[user_key("search_triggered")] = True

    user_query = st.text_input(
        "Enter your medical keyword to get matched policies...",
        key=user_key("user_query"),
        placeholder="e.g., 'Colonoscopy', 'Septoplasty', 'Steroid Injections'",
        on_change=handle_search_submit
    )

    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        search_clicked = st.button("🔍 Search", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        # Clear summary when clearing search
        st.session_state[user_key("last_chunks")] = []
        st.session_state[user_key("top_docs")] = []
        st.session_state[user_key("last_expansions")] = []
        st.session_state[user_key("selected_policy")] = None
        st.rerun()

    if (search_clicked or st.session_state.get(user_key("search_triggered"), False)) and user_query.strip():
        st.session_state[user_key("search_triggered")] = False
        query_text = user_query.strip()
        
        # Apply all security checks
        if SECURITY_ENABLED:
            is_valid, filtered_query, warning_msg = validate_user_input(
                query_text,
                domain="medical_policy",
                max_length=1000,
                check_injection=True,
                check_domain=True,
                check_trivial=True,
                check_gibberish=True,
                check_non_english=True,
                check_policy_relevance=True  #
            )
            
            if not is_valid:
                st.error(warning_msg)
                log_security_event("search_query_blocked", query_text, warning_msg)
                st.stop()
            
            query_text = filtered_query
        
        # Clear old summary when starting new search
        if st.session_state.get(user_key("selected_policy")):
            st.session_state[user_key("selected_policy")] = None
        
        with st.spinner("Searching policies..."):
            result = retrieve_top_contexts(
                user_query=query_text,
                db=st.session_state.db,
                llm_client=oai_client,
                top_k_final=5,
            )
            
            st.session_state[user_key("last_results")] = result
            st.session_state[user_key("last_chunks")] = result.get("ranked_results", [])
            st.session_state[user_key("top_docs")] = result.get("top_docs", [])
            st.session_state[user_key("last_expansions")] = result.get("variants", [])
        
        st.success("✅ Search complete!")

    # Display expanded queries
    user_expansions = st.session_state.get(user_key("last_expansions"), [])
    if user_expansions:
        exp_queries = ", ".join([f'"{q}"' for q in user_expansions[:5]])
        st.markdown(
            f"""
            <p style="
                color: #4a5568;
                font-size: 0.9rem;
                margin-top: 0.3rem;
                margin-bottom: 0.6rem;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            ">
                <strong>Expanded Queries:</strong> {exp_queries}
            </p>
            """,
            unsafe_allow_html=True
        )

    # Show matching policies
    user_last_chunks = st.session_state.get(user_key("last_chunks"), [])
    if user_last_chunks:
        n_chunks, n_docs, doc_ranked = compute_doc_stats(user_last_chunks)
        
        if n_chunks and n_docs:
            st.markdown(
                f"""
                <p style="
                    color: #4a5568;
                    font-size: 0.9rem;
                    margin-top: 0.3rem;
                    margin-bottom: 0.8rem;
                    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                ">
                    <strong>Retrieval Summary:</strong> {n_chunks} text chunks retrieved from {n_docs} policy documents.
                </p>
                """,
                unsafe_allow_html=True
            )

        top5 = st.session_state.get(user_key("top_docs"), [])[:5] or doc_ranked[:5]

        if top5:
            st.markdown(
                """
                <div style="
                    background-color: #e6f0fa;
                    border-left: 4px solid #2b6cb0;
                    padding: 4px 8px;
                    border-radius: 4px;
                    margin-top: 8px;
                    margin-bottom: 6px;
                ">
                    <h4 style="
                        margin: 0;
                        color: #1a202c;
                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                        font-size: 1.3rem;
                    ">
                        Matching Policies
                    </h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <p style="
                    color: #4a5568;
                    font-size: 0.9rem;
                    margin-top: 0.2rem;
                    margin-bottom: 0.8rem;
                    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                ">
                    Click <strong>Summarize</strong> beside a policy to view its summary below.
                </p>
                """,
                unsafe_allow_html=True
            )

            # Render policies
            render_policy_list(top5, user_last_chunks)


def render_policy_list(top_docs, user_last_chunks):
    """Render policy buttons and handle summarization"""
    for doc_id in top_docs:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"""
                <div style="
                    padding: 8px;
                    margin: 4px 0;
                    background-color: #f8f9fa;
                    border-left: 3px solid #48bb78;
                    border-radius: 4px;
                ">
                    <p style="
                        margin: 0;
                        font-size: 1.05rem;
                        font-weight: 600;
                        color: #2d3748;
                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                    ">
                        {doc_id}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            summarize_key = user_key(f"summarize_{doc_id}")
            
            if st.button("Summarize", key=summarize_key, use_container_width=True):
                st.session_state[user_key("selected_policy")] = doc_id

                # Get chunks for this policy
                matched_chunks = [
                    ch for ch in user_last_chunks
                    if isinstance(ch, dict)
                    and (
                        ch.get("metadata", {}).get("policy_id") == doc_id
                        or ch.get("metadata", {}).get("source") == doc_id
                        or ch.get("metadata", {}).get("doc_id") == doc_id
                        or ch.get("metadata", {}).get("file_name") == doc_id
                    )
                ]

                st.session_state[user_key(f"chunks_{doc_id}")] = matched_chunks

                # ALWAYS reset chat state when clicking Summarize
                # This ensures old conversations don't appear when viewing cached summary
                chat_key = user_key(f"chat_{doc_id}")
                st.session_state[chat_key] = {
                    "history": [],
                    "rate_limit_reached": False,
                    "chat_started": False,
                    "message_references": {},
                    "session_id": str(uuid.uuid4())  # Fresh session ID
                }

                # Check for cached summary ONLY (not conversations)
                username = st.session_state.auth["user_id"]
                cached_summary, cached_refs, cached_chunks = load_cached_summary(username, doc_id)
                
                if cached_summary and cached_summary.strip():
                    # Load from cache - SUMMARY ONLY
                    st.session_state[user_key(f"summary_{doc_id}")] = cached_summary
                    st.session_state[user_key(f"summary_refs_{doc_id}")] = cached_refs or {}
                    st.info("✅ Loaded summary from cache")
                elif matched_chunks:
                    # Generate new summary
                    with st.spinner(f"Summarizing {doc_id}..."):
                        from langchain_core.documents import Document

                        docs = [
                            Document(
                                page_content=ch.get("page_content", ""),
                                metadata=ch.get("metadata", {}),
                            )
                            for ch in matched_chunks
                            if ch.get("page_content")
                        ]
                        
                        result = summarize_policy_chunks(
                            retrieved_chunks=docs,
                            llm_client=oai_client,
                            llm_model=st.session_state.model_sel,
                        )

                        summary = result.get("summary", "")
                        refs = result.get("references", {})
                        
                        st.session_state[user_key(f"summary_{doc_id}")] = summary
                        st.session_state[user_key(f"summary_refs_{doc_id}")] = refs
                
                st.rerun()

    # Show summary for selected policy
    selected_policy = st.session_state.get(user_key("selected_policy"))
    if selected_policy:
        summary_text = st.session_state.get(user_key(f"summary_{selected_policy}"), "").strip()
        summary_refs = st.session_state.get(user_key(f"summary_refs_{selected_policy}"), {})

        if summary_text:
            with st.expander(f"Policy Summary: {selected_policy}", expanded=True):
                st.markdown(
                    "<p style='font-size:14px; color:gray; margin-top:0px; margin-bottom:8px;'>"
                    "<em>For Policy Review and Educational Use Only</em></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(summary_text)

                if summary_refs and isinstance(summary_refs, dict) and len(summary_refs) > 0:
                    grouped_refs = group_citations_by_document(summary_refs)
                    with st.expander(
                        f"Summary References ({len(summary_refs)} citations from {len(grouped_refs)} document(s))",
                        expanded=False,
                    ):
                        for doc_name, citations in grouped_refs.items():
                            citations_sorted = sorted(
                                citations, key=lambda x: int(x[0].strip("[]"))
                            )
                            st.markdown(f"**Policy Document:** {doc_name}")
                            cite_keys = [key for key, _ in citations_sorted]
                            pages = [page for _, page in citations_sorted]
                            st.markdown(f"- Citations: {', '.join(cite_keys)}")
                            st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                            st.markdown("")

            # "Start New Chat" button ALWAYS available
            st.divider()
            chat_key = user_key(f"chat_{selected_policy}")
            chat_started = st.session_state.get(chat_key, {}).get("chat_started", False)
            
            # Show appropriate button based on state
            if not chat_started:
                if st.button("💬 Start New Chat", key=user_key(f"start_chat_{selected_policy}"), use_container_width=True):
                    # Mark chat as started
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = {
                            "history": [],
                            "rate_limit_reached": False,
                            "chat_started": True,
                            "message_references": {},
                            "session_id": str(uuid.uuid4())
                        }
                    else:
                        st.session_state[chat_key]["chat_started"] = True
                    st.rerun()
            else:
                # Chat is active - show both chat interface and "Start New" button
                col_chat, col_new = st.columns([8, 2])
                with col_new:
                    if st.button("🔄 Start New", key=user_key(f"reset_chat_{selected_policy}"), 
                                use_container_width=True, help="Save current chat and start fresh"):
                        # Save current conversation to database
                        username = st.session_state.auth["user_id"]
                        current_history = st.session_state[chat_key].get("history", [])
                        
                        if current_history:
                            summary = st.session_state.get(user_key(f"summary_{selected_policy}"), "")
                            summary_refs = st.session_state.get(user_key(f"summary_refs_{selected_policy}"), {})
                            chunks = st.session_state.get(user_key(f"chunks_{selected_policy}"), [])
                            chat_refs = st.session_state[chat_key].get("message_references", {})
                            session_id = st.session_state[chat_key].get("session_id")
                            
                            # Save old conversation
                            save_conversation(username, selected_policy, summary, summary_refs, chunks, 
                                            current_history, chat_refs, session_id)
                            
                            st.success("✅ Previous conversation saved!")
                        
                        # Reset chat state with NEW session ID
                        st.session_state[chat_key] = {
                            "history": [],
                            "rate_limit_reached": False,
                            "chat_started": True,
                            "message_references": {},
                            "session_id": str(uuid.uuid4())  # New session ID for new conversation
                        }
                        time.sleep(0.5)
                        st.rerun()
                
                # Show chat interface
                render_chat_interface(selected_policy)


def render_conversation_view():
    """Render a READ-ONLY saved conversation view"""
    
    # Get the conversation to display
    conv = st.session_state.get("viewing_conversation")
    
    if not conv:
        st.error("Conversation not found!")
        if st.button("← Back to Search"):
            st.session_state.view_mode = "search"
            st.session_state.is_read_only = False
            st.rerun()
        return
    
    policy_name = conv["policy_name"]
    
    # Home button
    col1, col2 = st.columns([9, 1])
    with col1:
        st.info(f"Viewing saved conversation: {policy_name}")
    with col2:
        if st.button("🏠", key="home_btn", help="Return to Search", use_container_width=True):
            st.session_state.view_mode = "search"
            st.session_state.viewing_policy = None
            st.session_state.is_read_only = False
            del st.session_state["viewing_conversation"]
            st.rerun()
    
    # Display policy header
    st.markdown(
        f"""
        <div style="
            background-color: #e6f0fa;
            border-left: 4px solid #2b6cb0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
        ">
            <h4 style="
                margin: 0;
                color: #1a202c;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 1.3rem;
            ">
                {policy_name}
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Show summary
    if conv.get("summary"):
        with st.expander("Policy Summary", expanded=False):
            st.markdown(
                "<p style='font-size:14px; color:gray; margin-top:0px; margin-bottom:8px;'>"
                "<em>For Policy Review and Educational Use Only</em></p>",
                unsafe_allow_html=True,
            )
            st.markdown(conv["summary"])
            
            summary_refs = conv.get("summary_refs", {})
            if summary_refs and isinstance(summary_refs, dict) and len(summary_refs) > 0:
                grouped_refs = group_citations_by_document(summary_refs)
                with st.expander(
                    f"Summary References ({len(summary_refs)} citations from {len(grouped_refs)} document(s))",
                    expanded=False,
                ):
                    for doc_name, citations in grouped_refs.items():
                        citations_sorted = sorted(citations, key=lambda x: int(x[0].strip("[]")))
                        st.markdown(f"**Policy Document:** {doc_name}")
                        cite_keys = [key for key, _ in citations_sorted]
                        pages = [page for _, page in citations_sorted]
                        st.markdown(f"- Citations: {', '.join(cite_keys)}")
                        st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                        st.markdown("")
    
    st.divider()
    
    # Display READ-ONLY chat history
    st.markdown(
        """
        <div style="
            background-color: #e6f0fa;
            border-left: 4px solid #2b6cb0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
        ">
            <h4 style="
                margin: 0;
                color: #1a202c;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 1.3rem;
            ">
                Conversation History (Read-Only)
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    history = conv.get("history", [])
    message_refs = conv.get("message_references", {})
    
    if history:
        for msg_idx, msg in enumerate(history):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="
                            background-color: #4299e1; color: #ffffff;
                            padding: 12px 16px; border-radius: 18px;
                            max-width: 70%;
                            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                            font-size: 0.95rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                        <div style="
                            background-color: #f0f2f6; color: #1a202c;
                            padding: 12px 16px; border-radius: 18px;
                            max-width: 70%;
                            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                            font-size: 0.95rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show chat citations
                refs = message_refs.get(str(msg_idx), {})
                if isinstance(refs, dict) and len(refs) > 0:
                    grouped_refs = group_citations_by_document(refs)
                    with st.expander(
                        f"Chat References ({len(refs)} citations from {len(grouped_refs)} document(s))",
                        expanded=False,
                    ):
                        for doc_name, citations in grouped_refs.items():
                            citations_sorted = sorted(citations, key=lambda x: int(x[0].strip("[]")))
                            st.markdown(f"**Policy Document:** {doc_name}")
                            cite_keys = [key for key, _ in citations_sorted]
                            pages = [page for _, page in citations_sorted]
                            st.markdown(f"- Citations: {', '.join(cite_keys)}")
                            st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                            st.markdown("")
    
    # Show disabled input to indicate read-only mode
    st.text_input(
        "This is a saved conversation (Read-Only)",
        key="readonly_input",
        disabled=True,
        placeholder="Cannot add messages to saved conversations"
    )
    
    st.info("💡 To start a new conversation on this policy, go back to Search and select it again.")


def render_chat_interface(name):
    """Render interactive chat interface for a policy"""
    chat_key = user_key(f"chat_{name}")
    
    # Initialize if needed
    if chat_key not in st.session_state:
        st.session_state[chat_key] = {
            "history": [],
            "rate_limit_reached": False,
            "chat_started": False,
            "message_references": {},
            "session_id": str(uuid.uuid4())
        }

    st.markdown(
        """
        <div style="
            background-color: #e6f0fa;
            border-left: 4px solid #2b6cb0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
        ">
            <h4 style="
                margin: 0;
                color: #1a202c;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 1.3rem;
            ">
                Chat with Policy
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TOKEN LIMIT WARNING: Check and display if approaching limit
    try:
        from utils.rag import count_tokens, check_token_limit
        history = st.session_state[chat_key].get("history", [])
        
        if history:
            is_safe, token_count, token_warning = check_token_limit(
                history,
                model=st.session_state.model_sel,
                warning_threshold=100000,
                max_limit=120000
            )
            
            # Display token count for transparency
            if token_count > 50000:  # Only show after significant usage
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.caption(f"🔢 {token_count:,} tokens")
            
            # Display warning if approaching limit
            if token_warning:
                if is_safe:
                    st.warning(token_warning)
                else:
                    st.error(token_warning)
    except ImportError:
        pass  # Token counting not available

    # Display chat history
    history = st.session_state[chat_key].get("history", [])
    if history:
        for msg_idx, msg in enumerate(history):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="
                            background-color: #4299e1; color: #ffffff;
                            padding: 12px 16px; border-radius: 18px;
                            max-width: 70%;
                            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                            font-size: 0.95rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                        <div style="
                            background-color: #f0f2f6; color: #1a202c;
                            padding: 12px 16px; border-radius: 18px;
                            max-width: 70%;
                            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                            font-size: 0.95rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show chat citations
                refs = st.session_state[chat_key]["message_references"].get(msg_idx, {})
                
                # Debug: Show what we're trying to display
                if msg_idx in st.session_state[chat_key]["message_references"]:
                    print(f"📖 Displaying citations for message {msg_idx}: {len(refs)} citations")
                else:
                    print(f"⚠️ No citations found for message {msg_idx}")
                
                if isinstance(refs, dict) and len(refs) > 0:
                    grouped_refs = group_citations_by_document(refs)
                    with st.expander(
                        f"Chat References ({len(refs)} citations from {len(grouped_refs)} document(s))",
                        expanded=False,
                    ):
                        for doc_name, citations in grouped_refs.items():
                            citations_sorted = sorted(citations, key=lambda x: int(x[0].strip("[]")))
                            st.markdown(f"**Policy Document:** {doc_name}")
                            cite_keys = [key for key, _ in citations_sorted]
                            pages = [page for _, page in citations_sorted]
                            st.markdown(f"- Citations: {', '.join(cite_keys)}")
                            st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                            st.markdown("")    
    else:
        st.markdown(
            """
            <p style="
                color: #718096;
                font-size: 0.9rem;
                font-style: italic;
                text-align: center;
                margin-top: 16px;
                margin-bottom: 16px;
            ">
                No messages yet. Start the conversation by asking a question below.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # Chat input
    if st.session_state[chat_key]["rate_limit_reached"]:
        st.warning("⚠️ Rate limit reached. Please start a new chat session.")
        st.text_input(
            "Ask a question about this policy",
            key=user_key(f"chat_input_{name}_disabled"),
            disabled=True,
            placeholder="Rate limit reached - input disabled",
        )
    else:
        chat_input_key = user_key(f"chat_text_{name}")
        user_text = st.text_input(
            "Ask a question about this policy...",
            key=chat_input_key,
            placeholder="Type your question and press Enter or click Send..."
        )

        col_send, col_regen = st.columns([5, 1])
        
        with col_send:
            send_clicked = st.button("Send", key=user_key(f"send_{name}"), use_container_width=True)
        with col_regen:
            regen_clicked = st.button("🔄 Regenerate", key=user_key(f"regen_{name}"), help="Regenerate last response")

        if send_clicked and user_text.strip():
            try:
                # Apply all security checks INCLUDING injection detection
                if SECURITY_ENABLED:
                    is_valid, filtered_text, warning = validate_user_input(
                        user_text,
                        domain="medical_policy",
                        max_length=1000,
                        check_injection=True,  
                        check_trivial=True,
                        check_gibberish=True,
                        check_non_english=True,
                        check_policy_relevance=True  
                    )
                    
                    if not is_valid:
                        st.error(warning)
                        log_security_event("chat_input_blocked", user_text, warning)
                    else:
                        user_text = filtered_text
                        
                        depth_ok, depth_msg = check_conversation_depth(
                            st.session_state[chat_key]["history"],
                            max_turns=20
                        )
                        if not depth_ok:
                            st.warning(depth_msg)
                        else:
                            process_chat_question(name, user_text, chat_key)
                else:
                    process_chat_question(name, user_text, chat_key)
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                print(f"Chat error: {e}")
                traceback.print_exc()

        if regen_clicked:
            # Get the last user message before removing anything
            last_user = next(
                (msg["content"] for msg in reversed(st.session_state[chat_key]["history"]) if msg["role"] == "user"),
                None,
            )
            
            # Remove last assistant response AND its references
            if st.session_state[chat_key]["history"] and st.session_state[chat_key]["history"][-1]["role"] == "assistant":
                last_idx = len(st.session_state[chat_key]["history"]) - 1
                st.session_state[chat_key]["history"].pop()
                if last_idx in st.session_state[chat_key]["message_references"]:
                    del st.session_state[chat_key]["message_references"][last_idx]
                    print(f"Removed citation for message index {last_idx}")
            
            #  Remove last USER message (not assistant) - will be re-added by conversational_policy_qa
            if last_user:
                if st.session_state[chat_key]["history"] and st.session_state[chat_key]["history"][-1]["role"] == "user":
                    st.session_state[chat_key]["history"].pop()
                    print(f"Removed last user message for regeneration")

                process_chat_question(name, last_user, chat_key)


def process_chat_question(name, user_text, chat_key):
    """Process a chat question and save the conversation"""
    retrieved_chunks = st.session_state.get(user_key(f"chunks_{name}"), [])
    from langchain_core.documents import Document

    policy_docs = [
        Document(page_content=c.get("page_content", ""), metadata=c.get("metadata", {}))
        for c in retrieved_chunks if c.get("page_content")
    ]
    custom_prompt = st.session_state.get("system_prompt", None)

    with st.spinner("Thinking..."):
        answer, updated_history, rate_limit, references = conversational_policy_qa(
            retrieved_chunks=policy_docs,
            user_question=user_text,
            llm_client=oai_client,
            llm_model=st.session_state.model_sel,
            conversation_history=st.session_state[chat_key]["history"],
            custom_prompt=custom_prompt,
        )

        st.session_state[chat_key]["history"] = updated_history
        st.session_state[chat_key]["rate_limit_reached"] = rate_limit
        st.session_state[chat_key]["chat_started"] = True
        msg_idx = len(updated_history) - 1
        
        # Debug: Print what citations we're storing
        print(f" Storing citations for message {msg_idx}: {len(references)} citations")
        print(f"   Citations: {list(references.keys()) if references else 'None'}")
        
        st.session_state[chat_key]["message_references"][msg_idx] = references or {}

        # Save to database
        username = st.session_state.auth["user_id"]
        summary = st.session_state.get(user_key(f"summary_{name}"), "")
        summary_refs = st.session_state.get(user_key(f"summary_refs_{name}"), {})
        chat_refs = st.session_state[chat_key]["message_references"]
        session_id = st.session_state[chat_key].get("session_id")

        # Only save if there are actual messages
        if len(updated_history) > 0:
            save_conversation(username, name, summary, summary_refs, retrieved_chunks, 
                            updated_history, chat_refs, session_id)

    st.rerun()


# ===============================================
# Main Render Logic
# ===============================================

# Route based on view mode
if st.session_state.view_mode == "conversation":
    render_conversation_view()
else:
    render_search_view()


st.divider()
st.caption("✅ Conversations are automatically saved to database and persist across sessions.")
if SECURITY_ENABLED:
    st.caption("🛡️ Security features enabled: Prompt injection detection, trivial query filtering, gibberish detection, and domain validation active.")
