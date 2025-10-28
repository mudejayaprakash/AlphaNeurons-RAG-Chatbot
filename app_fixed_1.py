# app.py
# UM Policy RAG Assistant — COMPLETE FINAL VERSION
# All issues fixed: tabs switching, close buttons, Enter key, summary loading, citations

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
    conversational_policy_qa,
    group_citations_by_document
)

# Import security functions
try:
    from utils.security import validate_user_input, check_conversation_depth
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
    # Track loaded conversations for tabs
    if "loaded_conversations" not in ss:
        ss.loaded_conversations = {}  # {tab_id: policy_name}
    if "active_tab" not in ss:
        ss.active_tab = "main"

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
    
    # Table for storing conversations
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        policy_name TEXT,
        summary TEXT,
        summary_references TEXT,
        chunks TEXT,
        chat_references TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username),
        UNIQUE(username, policy_name)
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
    # Add preferred_name to users if missing
    try:
        c.execute("SELECT preferred_name FROM users LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE users ADD COLUMN preferred_name TEXT")

    # ---- Performance optimization ----
    # Create an index for faster per-user lookups
    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_conversations_username 
    ON conversations(username)
    """)
    conn.commit()
    conn.close()

def save_conversation(username, policy_name, summary, summary_refs, chunks, history, chat_refs=None):
    """Save or update a conversation in the database - using policy_name as unique identifier"""
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        
        # Check if conversation already exists
        c.execute("""
            SELECT id FROM conversations 
            WHERE username=? AND policy_name=?
        """, (username, policy_name))
        result = c.fetchone()
        
        chunks_json = json.dumps(chunks)
        summary_refs_json = json.dumps(summary_refs) if summary_refs else "{}"
        
        if result:
            # Update existing conversation
            conv_id = result[0]
            c.execute("""
                UPDATE conversations 
                SET summary=?, summary_references=?, chunks=?
                WHERE id=? AND username=?
            """, (summary, summary_refs_json, chunks_json, conv_id, username))
            
            # Delete old messages
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        else:
            # Create new conversation
            c.execute("""
                INSERT INTO conversations (username, policy_name, summary, summary_references, chunks)
                VALUES (?, ?, ?, ?, ?)
            """, (username, policy_name, summary, summary_refs_json, chunks_json))
            conv_id = c.lastrowid
        
        # Save chat messages
        if history:
            for msg in history:
                c.execute("""
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (?, ?, ?)
                """, (conv_id, msg.get("role"), msg.get("content")))
        
        # Save chat references if provided
        if chat_refs:
            chat_refs_json = json.dumps(chat_refs)
            c.execute("""
                UPDATE conversations 
                SET chat_references=?
                WHERE id=?
            """, (chat_refs_json, conv_id))
        
        conn.commit()
        print(f"✅ Saved conversation for policy '{policy_name}'")
    
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()

def load_user_conversations(username):
    """Load all conversations for a user from database"""
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    conversations = {}
    
    try:
        c.execute("""
            SELECT id, policy_name, summary, summary_references, chunks, chat_references
            FROM conversations
            WHERE username=?
            ORDER BY created_at DESC
        """, (username,))
        
        for row in c.fetchall():
            conv_id, policy_name, summary, summary_refs_json, chunks_json, chat_refs_json = row
            
            # Load messages for this conversation
            c.execute("""
                SELECT role, content FROM messages
                WHERE conversation_id=?
                ORDER BY timestamp ASC
            """, (conv_id,))
            
            messages = [{"role": role, "content": content} for role, content in c.fetchall()]
            
            # Parse JSON fields
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
            
            # ✅ FIXED #2: Store using policy_name as key, not index
            chat_key = user_key(f"chat_{policy_name}")
            conversations[chat_key] = {
                "history": messages,
                "chunks": chunks,
                "chat_started": len(messages) > 0,
                "policy_name": policy_name,
                "summary": summary or "",
                "summary_refs": summary_refs,
                "message_references": chat_refs
            }
    
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
st.set_page_config(page_title="PolicyMind", page_icon="🩺", layout="wide")

init_user_table()
ensure_session_state()

# ✅ FIXED: Session persistence - check query params for login
if "user" in st.query_params and not st.session_state.auth["logged_in"]:
    username = st.query_params["user"]
    # Try to restore session
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, preferred_name FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        st.session_state.auth = {"logged_in": True, "user_id": result[0], "preferred_name": result[1] or result[0]}
        # Load user's conversations
        loaded_convs = load_user_conversations(username)
        for chat_key, conv_data in loaded_convs.items():
            st.session_state[chat_key] = {
                "history": conv_data["history"],
                "chunks": conv_data["chunks"],
                "rate_limit_reached": False,
                "chat_started": safe_bool(conv_data["chat_started"]),
                "message_references": conv_data.get("message_references", {}),
                "summary": conv_data.get("summary", ""),
                "summary_refs": conv_data.get("summary_refs", {}),
            }
            
            policy_name = conv_data["policy_name"]
            st.session_state[user_key(f"summary_{policy_name}")] = conv_data.get("summary", "")
            st.session_state[user_key(f"summary_refs_{policy_name}")] = conv_data.get("summary_refs", {})
            st.session_state[user_key(f"chunks_{policy_name}")] = conv_data.get("chunks", [])

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
            help="Customize the assistant's behavior for question-answering.",
            height=200
        )

        st.divider()
        st.subheader("💬 Policy Conversations")
        
        # Collect all chat sessions that have been started
        policy_chats = []
        prefix = f"{uid}_chat_"
        for key in st.session_state.keys():
            if key.startswith(prefix) and isinstance(st.session_state[key], dict):
                if st.session_state[key].get("chat_started", False):
                    # ✅ FIXED #2: Extract policy name directly from key
                    policy_name = key.replace(prefix, "")
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
                        "name": policy_name,
                        "msg_count": msg_count,
                        "preview": preview
                    })
        
        if policy_chats:
            # Sort by most recent activity (those with more messages first)
            policy_chats.sort(key=lambda x: x["msg_count"], reverse=True)

            for chat in policy_chats:
                # Create button label with message count
                label = f"📄 {chat['name'][:40]}"
                if chat['msg_count'] > 0:
                    label += f" ({chat['msg_count']//2} msgs)"

                # Show preview as help text if available
                if st.button(label, key=f"sidebar-{chat['key']}", use_container_width=True,
                        help=chat['preview'] if chat['preview'] else "No messages yet"):

                    # ✅ FIXED #1: Load conversation and SWITCH to new tab
                    tab_id = f"prev_{chat['name']}"
                    st.session_state["loaded_conversations"][tab_id] = chat['name']
                    st.session_state["active_tab"] = tab_id  # Switch to this tab
                    
                    # Trigger rerun to display the tab
                    st.rerun()

        else:
            st.info("No conversations yet. Summarize a policy and start chatting!")
        
        # ---- Logout Button ----
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            # Save all conversations before logout
            username = st.session_state.auth["user_id"]
            prefix = f"{uid}_chat_"
            for key in list(st.session_state.keys()):
                if key.startswith(prefix) and isinstance(st.session_state[key], dict):
                    if st.session_state[key].get("chat_started", False):
                        policy_name = key.replace(prefix, "")
                        summary     = st.session_state.get(user_key(f"summary_{policy_name}"), "")
                        summary_refs= st.session_state.get(user_key(f"summary_refs_{policy_name}"), {})
                        chunks      = st.session_state.get(user_key(f"chunks_{policy_name}"), [])
                        history = st.session_state[key].get("history", [])
                        chat_refs = st.session_state[key].get("message_references", {})
                        save_conversation(username, policy_name, summary, summary_refs, chunks, history, chat_refs)

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
                # --- Clear old non-user session data on login ---
                for k in list(st.session_state.keys()):
                    if not k.startswith("auth"):
                        del st.session_state[k]

                # wipe any prior user's state
                reset_non_auth_state()
                st.session_state.auth = {"logged_in": True, "user_id": uid_input, "preferred_name": preferred_name}
                # ✅ Set query param for session persistence
                st.query_params["user"] = uid_input

                # Load user's conversations from database
                loaded_convs = load_user_conversations(uid_input)
                for chat_key, conv_data in loaded_convs.items():
                    st.session_state[chat_key] = {
                        "history": conv_data["history"],
                        "chunks": conv_data["chunks"],
                        "rate_limit_reached": False,
                        "chat_started": safe_bool(conv_data["chat_started"]),
                        "message_references": conv_data.get("message_references", {}),
                        "summary": conv_data.get("summary", ""),
                        "summary_refs": conv_data.get("summary_refs", {}),
                    }
                    
                    policy_name = conv_data["policy_name"]
                    st.session_state[user_key(f"summary_{policy_name}")] = conv_data.get("summary", "")
                    st.session_state[user_key(f"summary_refs_{policy_name}")] = conv_data.get("summary_refs", {})
                    st.session_state[user_key(f"chunks_{policy_name}")] = conv_data.get("chunks", [])

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
                else:
                    st.error(msg)
            else:
                st.warning("Please provide both username and password.")

    st.stop()


# ===============================================
# Main App (Post-Login)
# ===============================================

# ---- DB Health Check ----
if st.session_state.db is None:
    st.error(f"❌ Vector database not loaded. Error: {st.session_state.get('db_load_error', 'Unknown')}")
    st.stop()


# ✅ Tab structure with close buttons
def render_main_tab():
    """Render the main Search & Chat tab"""
    # ---- Search Interface ----
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
        """Handle search submission when Enter is pressed"""
        query = st.session_state.get(user_key("user_query"), "").strip()
        if query:
            st.session_state[user_key("search_triggered")] = True

    user_query = st.text_input(
        "Enter your medical question or keyword...",
        key=user_key("user_query"),
        placeholder="e.g., 'MRI coverage criteria'",
        on_change=handle_search_submit
    )

    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        search_clicked = st.button("🔍 Search", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state[user_key("last_chunks")] = []
        st.session_state[user_key("top_docs")] = []
        st.session_state[user_key("last_expansions")] = []
        st.session_state.pop(user_key("active_chat"), None)
        st.rerun()

    if (search_clicked or st.session_state.get(user_key("search_triggered"), False)) and user_query.strip():
        st.session_state[user_key("search_triggered")] = False
        query_text = user_query.strip()
        
        # SECURITY CHECK
        if SECURITY_ENABLED:
            is_valid, filtered_query, warning_msg = validate_user_input(
                query_text,
                domain="medical_policy",
                max_length=1000,
                check_injection=True,
                check_domain=True
            )
            
            if not is_valid:
                st.error(warning_msg)
                st.stop()
            
            query_text = filtered_query
        
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

    # Show matching policies and render them
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
                    Click <strong>🩺 Summarize</strong> beside a policy to view its summary below.
                </p>
                """,
                unsafe_allow_html=True
            )

            # Render policies
            render_policies(top5, user_last_chunks)


def render_previous_conversation_tab(policy_name, tab_id):
    """Render a previous conversation tab - shows summary AND chat with navigation buttons"""
    
    # ✅ FIXED #1: Add Back and Exit buttons
    col1, col2, col3 = st.columns([8, 1, 1])
    with col1:
        st.info("📄 Viewing previous conversation")
    with col2:
        if st.button("⬅ Back", key=f"back_{tab_id}", help="Return to Search & Chat (keeps this tab open)", use_container_width=True):
            # Switch to main tab but keep this tab open
            st.session_state["active_tab"] = "main"
            st.rerun()
    with col3:
        if st.button("✖ Exit", key=f"exit_{tab_id}", help="Close this tab and return to Search & Chat", use_container_width=True):
            # Remove from loaded conversations and switch to main
            st.session_state["loaded_conversations"].pop(tab_id, None)
            st.session_state["active_tab"] = "main"
            st.rerun()
    
    chat_key = user_key(f"chat_{policy_name}")
    
    # Check if conversation exists
    if chat_key not in st.session_state:
        st.error("Conversation not found!")
        return
    
    chat_data = st.session_state[chat_key]
    
    # Display policy name as header
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
    
    # Display summary if exists
    summary_text = st.session_state.get(user_key(f"summary_{policy_name}"), "").strip()
    summary_refs = st.session_state.get(user_key(f"summary_refs_{policy_name}"), {})
    
    if summary_text:
        with st.expander(f"Policy Summary: {policy_name}", expanded=True):
            st.markdown(
                "<p style='font-size:14px; color:gray; margin-top:0px; margin-bottom:8px;'>"
                "<em>For Policy Review and Educational Use Only</em></p>",
                unsafe_allow_html=True
            )
            st.markdown(summary_text)
            
            # Display summary references
            if summary_refs and isinstance(summary_refs, dict) and len(summary_refs) > 0:
                grouped_refs = group_citations_by_document(summary_refs)
                with st.expander(
                    f"Summary References ({len(summary_refs)} citations from {len(grouped_refs)} document(s))", 
                    expanded=False
                ):
                    for doc_name in sorted(grouped_refs.keys()):
                        citations = grouped_refs[doc_name]
                        citations_sorted = sorted(citations, key=lambda x: int(x[0].strip('[]')))
                        st.markdown(f"**Policy Document:** {doc_name}")
                        pages = [page for _, page in citations_sorted]
                        cite_keys = [key for key, _ in citations_sorted]
                        st.markdown(f"- Citations: {', '.join(cite_keys)}")
                        st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                        st.markdown("")
    
    # Display conversation
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
    
    # Display messages
    if chat_data["history"]:
        for msg_idx, msg in enumerate(chat_data["history"]):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="
                            background-color: #2b6cb0; color: white;
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
                
                # ✅ FIXED #5: Show chat citations properly - removed restrictive check
                refs = chat_data["message_references"].get(msg_idx, {})
                print(f"DEBUG: Message {msg_idx} refs: {refs}")  # Debug line
                if isinstance(refs, dict) and len(refs) > 0:
                    grouped_refs = group_citations_by_document(refs)
                    with st.expander(
                        f"Chat References ({len(refs)} citations from {len(grouped_refs)} document(s))",
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
    else:
        st.info("No messages in this conversation yet.")


def render_policies(top5, user_last_chunks):
    """Render policy list with summarize buttons - ONLY shows summary, NOT chat"""
    # ✅ FIXED #4: Removed chat display from this function
    selected_to_summarize = None
    
    # Check if user wants to view a specific policy
    if user_key("active_policy") in st.session_state:
        selected_to_summarize = st.session_state.get(user_key("active_policy"))

    # Display all policies with summarize buttons
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
            # ✅ FIXED #2: Use policy name as unique identifier
            if st.button("🩺 Summarize", key=user_key(f"summarize-{name}"), use_container_width=True):
                selected_to_summarize = name
                st.session_state[user_key("active_policy")] = name

    # If a policy is selected, display ONLY the summary
    if selected_to_summarize:
        name = selected_to_summarize

        # ✅ FIXED #2: Use policy name as unique identifier
        existing_summary = st.session_state.get(user_key(f"summary_{name}"), "")
        existing_summary_refs = st.session_state.get(user_key(f"summary_refs_{name}"), {})
        existing_chunks = st.session_state.get(user_key(f"chunks_{name}"), [])

        # If we have existing data, use it
        if existing_summary and existing_chunks:
            st.info(f"📄 Using existing summary for {name}.")
            summary_text = existing_summary
            summary_refs = existing_summary_refs
        else:
            # Prepare cache keys
            import hashlib
            cache_key = hashlib.md5(name.encode()).hexdigest()[:8]
            summary_key = user_key(f"summary_{cache_key}")
            summary_refs_key = user_key(f"summary_refs_{cache_key}")
            chunks_key = user_key(f"chunks_{cache_key}")

            # Extract chunks if needed
            if not st.session_state.get(user_key(f"chunks_{name}")):
                selected_chunks = []
                for c in user_last_chunks:
                    if not isinstance(c, dict):
                        continue
                    meta = c.get("metadata", {}) or {}
                    match_text = name.lower()
                    for field in ["source", "file_name", "policy_id", "doc_id"]:
                        if match_text in str(meta.get(field, "")).lower():
                            selected_chunks.append({
                                "page_content": c.get("page_content", ""),
                                "metadata": meta
                            })
                            break
                if not selected_chunks:
                    selected_chunks = st.session_state.get(user_key("last_chunks"), [])[:10]

                st.session_state[chunks_key] = selected_chunks
                st.session_state[user_key(f"chunks_{name}")] = selected_chunks

            # Check cache
            if st.session_state.get(summary_key) and st.session_state.get(summary_refs_key):
                st.info(f"📄 Using cached summary for {name}.")
                summary_text = st.session_state[summary_key]
                summary_refs = st.session_state[summary_refs_key]
                st.session_state[user_key(f"summary_{name}")] = summary_text
                st.session_state[user_key(f"summary_refs_{name}")] = summary_refs
            else:
                # Generate summary
                st.markdown("---")
                status = st.empty()
                status.markdown(f"**Summarizing {name}... Please wait.**")

                with st.spinner(f"Generating summary for {name}..."):
                    current_chunks = st.session_state.get(chunks_key, [])
                    if not current_chunks:
                        current_chunks = st.session_state.get(user_key(f"chunks_{name}"), [])
                    
                    if current_chunks:
                        summary_result = summarize_policy_chunks(
                            current_chunks,
                            llm_client=oai_client,
                            llm_model=st.session_state.model_sel
                        )
                        
                        if isinstance(summary_result, dict):
                            summary_text = summary_result.get("summary", "")
                            summary_refs = summary_result.get("references", {})
                        else:
                            summary_text = str(summary_result)
                            summary_refs = {}
                    else:
                        summary_text = "No policy content available for summarization."
                        summary_refs = {}

                    st.session_state[summary_key] = summary_text
                    st.session_state[summary_refs_key] = summary_refs
                    st.session_state[user_key(f"summary_{name}")] = summary_text
                    st.session_state[user_key(f"summary_refs_{name}")] = summary_refs

                    status.empty()
                    if current_chunks:
                        st.success(f"Summary for {name} generated!")

        # Display summary ONLY (no chat interface here)
        summary_text = st.session_state.get(user_key(f"summary_{name}"), "").strip()
        summary_refs = st.session_state.get(user_key(f"summary_refs_{name}"), {})

        if summary_text:
            with st.expander(f"Policy Summary: {name}", expanded=True):
                st.markdown(
                    "<p style='font-size:14px; color:gray; margin-top:0px; margin-bottom:8px;'>"
                    "<em>For Policy Review and Educational Use Only</em></p>",
                    unsafe_allow_html=True
                )
                st.markdown(summary_text)
                
                if summary_refs and isinstance(summary_refs, dict) and len(summary_refs) > 0:
                    grouped_refs = group_citations_by_document(summary_refs)
                    with st.expander(
                        f"Summary References ({len(summary_refs)} citations from {len(grouped_refs)} document(s))", 
                        expanded=False
                    ):
                        for doc_name in sorted(grouped_refs.keys()):
                            citations = grouped_refs[doc_name]
                            citations_sorted = sorted(citations, key=lambda x: int(x[0].strip('[]')))
                            st.markdown(f"**Policy Document:** {doc_name}")
                            pages = [page for _, page in citations_sorted]
                            cite_keys = [key for key, _ in citations_sorted]
                            st.markdown(f"- Citations: {', '.join(cite_keys)}")
                            st.markdown(f"- Pages: {', '.join(map(str, pages))}")
                            st.markdown("")
        
        # Chat interface section
        st.markdown("---")
        render_chat_interface(name)


def render_chat_interface(name):
    """Render chat interface for a policy"""
    chat_key = user_key(f"chat_{name}")
    
    # Initialize chat state if needed
    if chat_key not in st.session_state:
        st.session_state[chat_key] = {
            "history": [],
            "chat_started": False,
            "rate_limit_reached": False,
            "message_references": {}
        }
    
    # Header
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
                💬 Chat about this Policy
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display messages
    if st.session_state[chat_key]["history"]:
        for msg_idx, msg in enumerate(st.session_state[chat_key]["history"]):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="
                            background-color: #2b6cb0; color: white;
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
                if isinstance(refs, dict) and len(refs) > 0:
                    grouped_refs = group_citations_by_document(refs)
                    with st.expander(
                        f"Chat References ({len(refs)} citations from {len(grouped_refs)} document(s))",
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
        # ✅ FIXED #3: Remove on_change to prevent double submission
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
            regen_clicked = st.button("🔄", key=user_key(f"regen_{name}"), help="Regenerate last response")

        # ✅ FIXED #3: Only handle button click, not on_change
        if send_clicked and user_text.strip():
            try:
                # Security check
                if SECURITY_ENABLED:
                    is_valid, filtered_text, warning = validate_user_input(
                        user_text,
                        domain="medical_policy",
                        max_length=1000
                    )
                    
                    if not is_valid:
                        st.error(warning)
                    else:
                        user_text = filtered_text
                        
                        # Check conversation depth
                        depth_ok, depth_msg = check_conversation_depth(
                            st.session_state[chat_key]["history"],
                            max_turns=20
                        )
                        if not depth_ok:
                            st.warning(depth_msg)
                        else:
                            # Process question
                            process_chat_question(name, user_text, chat_key)
                else:
                    # No security - process directly
                    process_chat_question(name, user_text, chat_key)
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                print(f"Chat error: {e}")
                traceback.print_exc()

        # Handle regenerate
        if regen_clicked:
            last_user = next(
                (msg["content"] for msg in reversed(st.session_state[chat_key]["history"]) if msg["role"] == "user"),
                None,
            )

            if last_user:
                if st.session_state[chat_key]["history"] and st.session_state[chat_key]["history"][-1]["role"] == "assistant":
                    st.session_state[chat_key]["history"].pop()

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
        
        # Store references properly
        st.session_state[chat_key]["message_references"][msg_idx] = references or {}

        username = st.session_state.auth["user_id"]
        summary = st.session_state.get(user_key(f"summary_{name}"), "")
        summary_refs = st.session_state.get(user_key(f"summary_refs_{name}"), {})
        chat_refs = st.session_state[chat_key]["message_references"]

        save_conversation(username, name, summary, summary_refs, retrieved_chunks, updated_history, chat_refs)

    st.rerun()


# ✅ Render tabs dynamically
tab_names = ["🔍 Search & Chat"]
tab_keys = ["main"]

# Add tabs for loaded conversations
for tab_id, policy_name in st.session_state["loaded_conversations"].items():
    tab_names.append(f"📄 {policy_name[:30]}...")  # Truncate long names
    tab_keys.append(tab_id)

# Create tabs
if len(tab_names) == 1:
    # Only main tab
    render_main_tab()
else:
    # Multiple tabs - get active tab index
    active_tab_key = st.session_state.get("active_tab", "main")
    if active_tab_key not in tab_keys:
        active_tab_key = "main"
        st.session_state["active_tab"] = "main"
    
    # ✅ FIXED #1: Set default to active tab
    default_index = tab_keys.index(active_tab_key) if active_tab_key in tab_keys else 0
    
    tabs = st.tabs(tab_names)
    
    for idx, (tab, tab_key) in enumerate(zip(tabs, tab_keys)):
        # Only render the selected tab
        if idx == default_index:
            with tab:
                if tab_key == "main":
                    render_main_tab()
                else:
                    # Previous conversation tab
                    policy_name = st.session_state["loaded_conversations"][tab_key]
                    render_previous_conversation_tab(policy_name, tab_key)


st.divider()
st.caption("✅ Conversations are automatically saved to database and persist across sessions.")
if SECURITY_ENABLED:
    st.caption("🛡️ Security features enabled: Prompt injection detection and domain validation active.")
