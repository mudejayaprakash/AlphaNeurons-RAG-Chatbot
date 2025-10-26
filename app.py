# app.py
# UM Policy RAG Assistant — final, robust version
# - Auth (login + register to SQLite)
# - One-click Search (expands, retrieves, re-ranks)
# - Shows expanded queries, retrieval summary, top-5 policy docs (selectable)
# - Summarizes selected policy, then starts chat with memory + regenerate
# - Clean error handling (rate limits, missing DB, etc.)

import os
import time
import uuid
import sqlite3
import traceback
from collections import defaultdict

import streamlit as st

# ---- Import your shared OpenAI client and RAG helpers ----
from utils.config import openai_client as oai_client
from utils.rag import (
    load_vector_db, 
    expand_query,            
    semantic_search,        
    retrieve_top_contexts,   
    summarize_policy_chunks,
    conversational_policy_qa
)

# =========================
# Session / Auth Utilities
# =========================
DB_PATH = "chat_app.db"

def ensure_session_state():
    ss = st.session_state
    if "auth" not in ss:
        ss.auth = {"logged_in": False, "user_id": None}
    if "db" not in ss:
        try:
            ss.db = load_vector_db()
        except Exception as e:
            ss.db = None
            ss.db_load_error = str(e)
    if "model_sel" not in ss:
        ss.model_sel = "gpt-4o-mini"
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


def init_user_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    
    # Table for storing conversations
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        policy_index INTEGER,
        policy_name TEXT,
        summary TEXT,
        chunks TEXT,
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
    
    conn.commit()
    conn.close()

def save_conversation(username, policy_index, policy_name, summary, chunks, history):
    """Save or update a conversation in the database"""
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if conversation already exists
    c.execute("""
        SELECT id FROM conversations 
        WHERE username=? AND policy_index=? AND policy_name=?
    """, (username, policy_index, policy_name))
    result = c.fetchone()
    
    chunks_json = json.dumps(chunks)
    
    if result:
        # Update existing conversation
        conv_id = result[0]
        c.execute("""
            UPDATE conversations 
            SET summary=?, chunks=?
            WHERE id=?
        """, (summary, chunks_json, conv_id))
        
        # Delete old messages
        c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    else:
        # Create new conversation
        c.execute("""
            INSERT INTO conversations (username, policy_index, policy_name, summary, chunks)
            VALUES (?, ?, ?, ?, ?)
        """, (username, policy_index, policy_name, summary, chunks_json))
        conv_id = c.lastrowid
    
    # Save messages
    for msg in history:
        c.execute("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
        """, (conv_id, msg["role"], msg["content"]))
    
    conn.commit()
    conn.close()

def load_user_conversations(username):
    """Load all conversations for a user from database"""
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT id, policy_index, policy_name, summary, chunks
        FROM conversations
        WHERE username=?
        ORDER BY created_at DESC
    """, (username,))
    
    conversations = {}
    for row in c.fetchall():
        conv_id, policy_index, policy_name, summary, chunks_json = row
        
        # Load messages for this conversation
        c.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id=?
            ORDER BY timestamp ASC
        """, (conv_id,))
        
        history = [{"role": role, "content": content} for role, content in c.fetchall()]
        chunks = json.loads(chunks_json) if chunks_json else []
        
        # Store in session state format
        chat_key = f"chat_{policy_index}_{policy_name}"
        conversations[chat_key] = {
            "policy_index": policy_index,
            "policy_name": policy_name,
            "summary": summary,
            "chunks": chunks,
            "history": history,
            "chat_started": len(history) > 0,
            "rate_limit_reached": False
        }
    
    conn.close()
    return conversations

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def validate_login(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    if result:
        return True, result[0]
    return False, None

def get_user_id(username):
    """Get user ID from username"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def new_chat_session(title="New chat", policy_blob=None):
    sid = str(uuid.uuid4())[:8]
    st.session_state.conversations[sid] = {
        "title": title,
        "messages": [],
        "created_at": time.time(),
        "policy_context": policy_blob or {},
    }
    st.session_state.current_session_id = sid
    return sid

def get_current_conv():
    sid = st.session_state.current_session_id
    if not sid:
        return None, None
    return sid, st.session_state.conversations.get(sid)

# =========================
# Stats / Display Utilities
# =========================
def compute_doc_stats(ranked_chunks):
    """
    Expects a list of dicts with keys: 'score', 'page_content', 'metadata'.
    Supports 'policy_id' (preferred), or falls back to 'source'/'doc_id'/'file_name'.
    """
    if not ranked_chunks:
        return 0, 0, []

    by_doc = defaultdict(int)
    for c in ranked_chunks:
        if not isinstance(c, dict):
            continue
        meta = c.get("metadata", {}) or {}
        name = (
            meta.get("policy_id")
            or meta.get("source")
            or meta.get("doc_id")
            or meta.get("file_name")
            or "Unknown Document"
        )
        by_doc[name] += 1

    n_chunks = len(ranked_chunks)
    n_docs = len(by_doc)
    top_docs = [k for k, _ in sorted(by_doc.items(), key=lambda kv: kv[1], reverse=True)]
    return n_chunks, n_docs, top_docs

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="UM RAG Assistant", page_icon="🩺", layout="wide")
ensure_session_state()
init_user_table()

# ---- Session Persistence ----
# Check if user is already logged in via query params
query_params = st.query_params
if not st.session_state.auth["logged_in"]:
    if "user" in query_params:
        username = query_params["user"]
        # Auto-login if valid user
        if get_user_id(username):
            st.session_state.auth = {"logged_in": True, "user_id": username}
            # Load user's conversations from database
            loaded_convs = load_user_conversations(username)
            for chat_key, conv_data in loaded_convs.items():
                st.session_state[chat_key] = {
                    "history": conv_data["history"],
                    "chunks": conv_data["chunks"],
                    "rate_limit_reached": False,
                    "chat_started": conv_data["chat_started"]
                }
                # Also store summary if available
                if conv_data["summary"]:
                    policy_index = conv_data["policy_index"]
                    st.session_state[f"summary_{policy_index}"] = conv_data["summary"]
                    st.session_state[f"chunks_{policy_index}"] = conv_data["chunks"]
else:
    # User is logged in, ensure query param is set
    if "user" not in query_params or query_params["user"] != st.session_state.auth["user_id"]:
        st.query_params["user"] = st.session_state.auth["user_id"]

# ---- Sidebar (visible only after login) ----
if st.session_state.auth["logged_in"]:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; color: #2b6cb0;">👋 Hi, {st.session_state.auth['user_id'].title()}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("⚙️ Settings")
        st.session_state.model_sel = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "o1-preview"],
            index=0
        )
        # --- Custom System Prompt ---
        default_prompt = (
            "You are a helpful assistant specializing in medical policy and utilization management. "
            "Answer concisely, accurately, and based on the provided policy context when available."
        )

        st.session_state.system_prompt = st.text_area(
            "System Prompt (for Q&A only)",
            value=st.session_state.get("system_prompt", default_prompt),
            help="Customize the assistant’s behavior for question-answering.",
            height=200
        )

        st.divider()
        st.subheader("💬 Policy Conversations")
        
        # Collect all chat sessions that have been started
        policy_chats = []
        for key in st.session_state.keys():
            if key.startswith("chat_") and isinstance(st.session_state[key], dict):
                if st.session_state[key].get("chat_started", False):
                    # Extract policy info from key (format: chat_{i}_{name})
                    parts = key.split("_", 2)
                    if len(parts) == 3:
                        policy_i = int(parts[1])
                        policy_name = parts[2]
                        history = st.session_state[key].get("history", [])
                        msg_count = len(history)
                        
                        # Get first user message for preview
                        preview = ""
                        for msg in history:
                            if msg["role"] == "user":
                                preview = msg["content"][:50]
                                if len(msg["content"]) > 50:
                                    preview += "..."
                                break
                        
                        policy_chats.append({
                            "key": key,
                            "i": policy_i,
                            "name": policy_name,
                            "msg_count": msg_count,
                            "preview": preview
                        })
        
        if policy_chats:
            # Sort by most recent activity (those with more messages first)
            policy_chats.sort(key=lambda x: x["msg_count"], reverse=True)
            
            for chat in policy_chats:
                # Create button label with message count
                label = f"📄 {chat['name']}"
                if chat['msg_count'] > 0:
                    label += f" ({chat['msg_count']//2} msgs)"
                
                # Show preview as help text if available
                if st.button(label, key=f"sidebar-{chat['key']}", 
                           use_container_width=True,
                           help=chat['preview'] if chat['preview'] else "No messages yet"):
                    # Activate this chat
                    st.session_state["active_chat"] = (chat['i'], chat['name'])
                    st.rerun()
        else:
            st.info("No conversations yet. Summarize a policy and start chatting!")
        
        # ---- Logout Button ----
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            # Save all conversations before logout
            username = st.session_state.auth["user_id"]
            for key in st.session_state.keys():
                if key.startswith("chat_") and isinstance(st.session_state[key], dict):
                    if st.session_state[key].get("chat_started", False):
                        parts = key.split("_", 2)
                        if len(parts) == 3:
                            policy_i = int(parts[1])
                            policy_name = parts[2]
                            summary = st.session_state.get(f"summary_{policy_i}", "")
                            chunks = st.session_state.get(f"chunks_{policy_i}", [])
                            history = st.session_state[key].get("history", [])
                            save_conversation(username, policy_i, policy_name, summary, chunks, history)
            
            # Clear session
            st.session_state.auth = {"logged_in": False, "user_id": None}
            st.query_params.clear()
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()


st.markdown(
    """
    <div style="
        background-color: #e0f7f9;     /* soft blue-gray background */
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
        uid = st.text_input("Username", key="login_uid")
        pwd = st.text_input("Password", key="login_pwd", type="password")
        if st.button("Login"):
            success, user_id = validate_login(uid, pwd)
            if success:
                st.session_state.auth = {"logged_in": True, "user_id": uid}
                # Set query param for session persistence
                st.query_params["user"] = uid
                # Load user's conversations from database
                loaded_convs = load_user_conversations(uid)
                for chat_key, conv_data in loaded_convs.items():
                    st.session_state[chat_key] = {
                        "history": conv_data["history"],
                        "chunks": conv_data["chunks"],
                        "rate_limit_reached": False,
                        "chat_started": conv_data["chat_started"]
                    }
                    # Also store summary if available
                    if conv_data["summary"]:
                        policy_index = conv_data["policy_index"]
                        st.session_state[f"summary_{policy_index}"] = conv_data["summary"]
                        st.session_state[f"chunks_{policy_index}"] = conv_data["chunks"]
                st.success(f"Welcome back, {uid}! Your conversations have been loaded.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab2:
        st.subheader("Register")
        new_uid = st.text_input("Choose Username", key="reg_uid")
        new_pwd = st.text_input("Choose Password", key="reg_pwd", type="password")
        if st.button("Create Account"):
            if not new_uid or not new_pwd:
                st.warning("Please enter both username and password.")
            else:
                ok, msg = register_user(new_uid, new_pwd)
                st.success(msg) if ok else st.error(msg)
    st.stop()

# ---- Search Section ----

# 💠 Styled header for "Search for Relevant Policies"
st.markdown(
    """
    <div style="
        background-color: #e6f0fa;        /* soft blue background */
        border-left: 4px solid #2b6cb0;   /* blue accent bar */
        padding: 3px 8px;                /* reduced padding for compact look */
        border-radius: 4px;
        margin-top: 6px;
        margin-bottom: 8px;
    ">
        <h3 style="
            margin: 0;
            color: #1a202c;
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            font-size: 1.3rem;            /* slightly larger font */
            font-weight: 600;
        ">
            Search for Relevant Policies
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

query = st.text_input("Enter a condition, procedure, or topic (e.g., 'Septoplasty','Colonoscopy','Epidural Steroid Injections'):", key="search_query")

if st.button("Search"):
    if not query.strip():
        st.error("Please enter a search query.")
        st.stop()

    if st.session_state.db is None:
        st.error("Vector DB not loaded. Rebuild embeddings or fix PERSIST_DIR.")
        st.stop()

    with st.spinner("Searching the Policy knowledge base..."):
        try:
            results = retrieve_top_contexts(
                user_query=query,
                db=st.session_state.db,
                llm_client=oai_client,
                top_k_final=5,     # show top-5 docs in UI
                max_c=100          # re-rank + cap to top-100 chunks
            )
            st.session_state.last_results = results
            st.session_state.last_expansions = results.get("variants", [])
            st.session_state.last_chunks = results.get("ranked_results", [])
            st.session_state.top_docs = results.get("top_docs", [])
            st.success("Search complete!")
        except Exception as e:
            st.error(f"Search failed: {e}")
            print(traceback.format_exc())

# ---- Expanded Queries ----
if st.session_state.last_expansions:
    exp_queries = ", ".join(st.session_state.last_expansions)
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

# ---- Retrieval Summary + Top-5 ----
if st.session_state.last_chunks:
    n_chunks, n_docs, doc_ranked = compute_doc_stats(st.session_state.last_chunks)

    # ---- Retrieval Summary ----
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
    else:
        st.markdown(
            """
            <p style="
                color: #4a5568;
                font-size: 0.9rem;
                margin-top: 0.3rem;
                margin-bottom: 0.8rem;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            ">
                No chunks retrieved from the vector database.
            </p>
            """,
            unsafe_allow_html=True
        )

    # Prefer aggregated top_docs from retrieval; fallback to doc_ranked
    top5 = st.session_state.top_docs[:5] if st.session_state.top_docs else doc_ranked[:5]

    if top5:
        # 💠 Section Header
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

        # Instruction line
        st.markdown(
            """
            <p style="
                color: #4a5568;
                font-size: 0.9rem;
                margin-top: 0.2rem;
                margin-bottom: 0.8rem;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            ">
                Click <strong>🩺 Summarize</strong> beside a policy to view its summary below.
            </p>
            """,
            unsafe_allow_html=True
        )

        # ---- Policy List with Summarize Buttons ----
        selected_to_summarize = None
        
        # Check if there's an active chat to auto-select that policy
        if "active_chat" in st.session_state:
            selected_to_summarize = st.session_state["active_chat"]

        for i, name in enumerate(top5, 1):
            c1, c2 = st.columns([6, 2])
            with c1:
                st.markdown(
                    f"""
                    <p style="
                        color: #1a202c;
                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                        font-size: 0.98rem;
                        font-weight: 500;
                        margin: 6px 0;
                    ">
                        {i}. {name}
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🩺 Summarize", key=f"summarize-{i}", use_container_width=True):
                    selected_to_summarize = (i, name)
                    st.session_state["active_chat"] = (i, name)

        # ---- After policy list: handle summarization ----
        if selected_to_summarize:
            i, name = selected_to_summarize
            
            # Extract chunks for this policy (needed for both summary and chat)
            selected_chunks = []
            for c in st.session_state.last_chunks:
                if not isinstance(c, dict):
                    continue
                meta = c.get("metadata", {}) or {}
                src = (
                    meta.get("policy_id")
                    or meta.get("source")
                    or meta.get("doc_id")
                    or meta.get("file_name")
                )
                if src == name:
                    selected_chunks.append({
                        "page_content": c.get("page_content", ""),
                        "metadata": meta
                    })
            
            # Store chunks in session state for chat use
            st.session_state[f"chunks_{i}"] = selected_chunks

            if f"summary_{i}" in st.session_state:
                summary = st.session_state[f"summary_{i}"]
                st.info(f"📄 Using cached summary for {name}.")
            else:
                st.markdown("---")
                status = st.empty()
                status.markdown(f"**Summarizing {name}... Please wait.**")

                with st.spinner(f"Generating summary for {name}..."):
                    from langchain_core.documents import Document
                    policy_docs = [
                        Document(page_content=c.get("page_content", ""), metadata=c.get("metadata", {}))
                        for c in selected_chunks
                    ]

                    summary = summarize_policy_chunks(
                        retrieved_chunks=policy_docs,
                        llm_client=oai_client,
                        llm_model=st.session_state.model_sel,
                    )

                    st.session_state[f"summary_{i}"] = summary
                    status.empty()
                    st.success(f"Summary for {name} generated!")

            # ---- Display summary ----
            st.markdown(
                f"""
                <div style="
                    background-color: #e6f0fa;
                    border-left: 4px solid #2b6cb0;
                    padding: 6px 10px;
                    border-radius: 4px;
                    margin-top: 16px;
                    margin-bottom: 0;
                ">
                    <h4 style="
                        margin: 0;
                        color: #1a202c;
                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                        font-size: 1.1rem; 
                        font-weight: 600;
                    ">
                        Summary of {name}
                    </h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size:16px; color:gray; margin-top:4px; margin-bottom:8px;'>"
                "<em>For Policy Review and Educational Use Only</em></p>",
                unsafe_allow_html=True
            )
            summary_text = st.session_state.get(f"summary_{i}", "").strip()

            if summary_text:
                st.markdown(
                    f"<div style='margin-top:6px;margin-bottom:12px;'>", 
                    unsafe_allow_html=True
                )
                st.markdown(summary_text)  # renders bold, lists, etc.
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No summary generated yet.")

            # ---- Chat Section ----
            # Get chunks from session state
            policy_chunks = st.session_state.get(f"chunks_{i}", [])
            
            # Initialize chat for this policy if not already done
            chat_key = f"chat_{i}_{name}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = {
                    "history": [],
                    "chunks": policy_chunks,
                    "rate_limit_reached": False,
                    "chat_started": False
                }
            
            # Show "Start New Chat" button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("💬 Start New Chat", key=f"start_chat_{i}", use_container_width=True):
                    # Reset chat session
                    st.session_state[chat_key]["history"] = []
                    st.session_state[chat_key]["chunks"] = policy_chunks
                    st.session_state[chat_key]["rate_limit_reached"] = False
                    st.session_state[chat_key]["chat_started"] = True
                    # Keep this policy active so it shows after rerun
                    st.session_state["active_chat"] = (i, name)
                    st.rerun()
            
            # Show conversation section if chat has been started
            if st.session_state[chat_key].get("chat_started", False):
                # Display Conversation Section
                st.markdown(
                    """
                    <div style="
                        background-color: #e6f0fa;
                        border-left: 4px solid #2b6cb0;
                        padding: 4px 8px;
                        border-radius: 4px;
                        margin-top: 16px;
                        margin-bottom: 6px;
                    ">
                        <h4 style="
                            margin: 0;
                            color: #1a202c;
                            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                            font-size: 1.3rem;
                        ">
                            Conversation
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display conversation history
                if st.session_state[chat_key]["history"]:
                    for msg in st.session_state[chat_key]["history"]:
                        if msg["role"] == "user":
                            st.markdown(
                                f"""
                                <div style="
                                    display: flex;
                                    justify-content: flex-end;
                                    margin-bottom: 12px;
                                ">
                                    <div style="
                                        background-color: #2b6cb0;
                                        color: white;
                                        padding: 12px 16px;
                                        border-radius: 18px;
                                        max-width: 70%;
                                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                                        font-size: 0.95rem;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    ">
                                        {msg["content"]}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        elif msg["role"] == "assistant":
                            st.markdown(
                                f"""
                                <div style="
                                    display: flex;
                                    justify-content: flex-start;
                                    margin-bottom: 12px;
                                ">
                                    <div style="
                                        background-color: #f0f2f6;
                                        color: #1a202c;
                                        padding: 12px 16px;
                                        border-radius: 18px;
                                        max-width: 70%;
                                        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                                        font-size: 0.95rem;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    ">
                                        {msg["content"]}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
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
                        unsafe_allow_html=True
                    )
                
                # Chat input section
                if st.session_state[chat_key]["rate_limit_reached"]:
                    st.warning("⚠️ Rate limit reached. Please start a new chat session.")
                    st.text_input(
                        "Ask a question about this policy",
                        key=f"chat_input_{i}",
                        disabled=True,
                        placeholder="Rate limit reached - input disabled"
                    )
                else:
                    user_question = st.text_input(
                        "Ask a question about this policy",
                        key=f"chat_input_{i}",
                        placeholder="Type your question here..."
                    )
                    
                    if st.button("Send", key=f"send_{i}"):
                        if user_question.strip():
                            with st.spinner("Thinking..."):
                                from langchain_core.documents import Document
                                policy_docs = [
                                    Document(
                                        page_content=c.get("page_content", ""),
                                        metadata=c.get("metadata", {})
                                    )
                                    for c in st.session_state[chat_key]["chunks"]
                                ]
                                
                                custom_prompt = st.session_state.get("system_prompt", None)
                                
                                answer, updated_history, rate_limit = conversational_policy_qa(
                                    retrieved_chunks=policy_docs,
                                    user_question=user_question,
                                    llm_client=oai_client,
                                    llm_model=st.session_state.model_sel,
                                    conversation_history=st.session_state[chat_key]["history"],
                                    custom_prompt=custom_prompt
                                )
                                
                                st.session_state[chat_key]["history"] = updated_history
                                st.session_state[chat_key]["rate_limit_reached"] = rate_limit
                                st.session_state["active_chat"] = (i, name)
                                
                                # Auto-save conversation to database
                                username = st.session_state.auth["user_id"]
                                summary = st.session_state.get(f"summary_{i}", "")
                                chunks = st.session_state.get(f"chunks_{i}", [])
                                save_conversation(username, i, name, summary, chunks, updated_history)
                                
                                st.rerun()
                        else:
                            st.error("Please enter a question.")


st.divider()
st.caption("✅ Conversations are automatically saved to database and persist across sessions.")