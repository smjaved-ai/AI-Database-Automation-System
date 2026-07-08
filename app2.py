"""
Project 1: Agentic AI / RAG-Based Natural Language MongoDB Query System
Uses: MongoDB Atlas (sample_mflix), Google Gemini (free), Streamlit
"""

import streamlit as st
import pymongo
import google.generativeai as genai
import json
import re
from bson import ObjectId
import datetime

# ─── Hardcoded Credentials (replace with your actual values) ────────────────────
MONGO_URI = "YOUR_MONGODB_ATLAS_URI_HERE"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI MongoDB Query System", page_icon="🤖", layout="wide")

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTab [data-baseweb="tab"] { font-size: 18px; font-weight: bold; }
    .query-box { background: #1e1e2e; color: #cdd6f4; padding: 12px;
                 border-radius: 8px; font-family: monospace; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Info Only ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("**Project 1 — AI MongoDB Query System**")
    st.markdown("Ask questions in plain English and the AI generates and runs the MongoDB query for you.")
    st.markdown("---")
    st.markdown("**Database:** MongoDB Atlas `sample_mflix`")
    st.markdown("**AI Model:** Google Gemini 1.5 Flash")

mongo_uri = MONGO_URI
gemini_key = GEMINI_API_KEY

# ─── Helpers ────────────────────────────────────────────────────────────────────

def json_safe(obj):
    """Make MongoDB results JSON-serialisable."""
    if isinstance(obj, list):
        return [json_safe(i) for i in obj]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return obj


@st.cache_resource(show_spinner="Connecting to MongoDB…")
def get_mongo_client(uri: str):
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")      # will raise if connection fails
    return client


def build_schema_context(client, db_name="sample_mflix"):
    """RAG context: collect field names + sample values for each collection."""
    db = client[db_name]
    context_parts = []
    for coll_name in db.list_collection_names():
        sample = list(db[coll_name].find({}, {"_id": 0}).limit(2))
        if sample:
            fields = list(sample[0].keys())
            context_parts.append(
                f"Collection: `{coll_name}`\n"
                f"Fields: {fields}\n"
                f"Sample doc: {json.dumps(json_safe(sample[0]), default=str)[:400]}"
            )
    return "\n\n".join(context_parts)


def ask_gemini(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


def extract_python_query(text: str) -> str:
    """Pull out the Python/PyMongo expression from the AI reply."""
    # look for fenced code block first
    fenced = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if fenced:
        return fenced.group(1).strip()
    # fall back to first line that looks like a PyMongo call
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("db.") or line.startswith("collection"):
            return line
    return text.strip()


def execute_query(client, query_code: str, db_name="sample_mflix"):
    """
    Safely execute a generated PyMongo query.
    Only find / aggregate / count_documents are allowed.
    """
    db = client[db_name]
    allowed_ops = ("find", "aggregate", "count_documents", "distinct")
    if not any(op in query_code for op in allowed_ops):
        return None, "Only read operations (find / aggregate / count_documents) are permitted."

    # Provide `db` in the exec namespace so generated code can reference it
    local_ns = {"db": db}
    try:
        exec(f"_result = {query_code}", local_ns)   # noqa: S102
        result = local_ns.get("_result")
        if hasattr(result, "__iter__") and not isinstance(result, (dict, str, int, float)):
            return list(result)[:50], None           # cap at 50 rows
        return result, None
    except Exception as exc:
        return None, str(exc)


# ─── Initialise conversation memory ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role": ..., "content": ...}

# ─── Main Tab ───────────────────────────────────────────────────────────────────
tab1, = st.tabs(["Process1"])

with tab1:
    st.title("🤖 AI-Powered MongoDB Query System")
    st.caption("Ask questions in plain English — the AI generates and runs the MongoDB query for you.")

    # ── Initialise clients ───────────────────────────────────────────────────────
    try:
        client = get_mongo_client(mongo_uri)
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()

    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"❌ Gemini initialisation failed: {e}")
        st.stop()

    st.success("✅ Connected to MongoDB Atlas  |  Gemini ready")

    # ── Build / cache RAG schema context ────────────────────────────────────────
    with st.spinner("📚 Loading schema context (RAG)…"):
        schema_ctx = build_schema_context(client)

    with st.expander("📋 View RAG Schema Context"):
        st.text(schema_ctx)

    st.markdown("---")

    # ── Chat history display ─────────────────────────────────────────────────────
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── User input ───────────────────────────────────────────────────────────────
    user_prompt = st.chat_input("Ask a question about movies…  e.g. Show top 10 movies by IMDb rating")

    if user_prompt:
        # Show user message
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # ── Build conversation context for follow-up support ─────────────────────
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.history[-6:]   # last 6 turns
        )

        # ── Ask Gemini to generate a PyMongo query ───────────────────────────────
        ai_prompt = f"""
You are an expert MongoDB query generator.

DATABASE: sample_mflix (MongoDB Atlas sample dataset)

SCHEMA CONTEXT (RAG):
{schema_ctx}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_prompt}

Instructions:
1. Identify the correct collection.
2. Write a single PyMongo expression (Python syntax using `db`).
3. Always add `.limit(20)` unless the user asks for a specific number ≤ 50.
4. Return ONLY the PyMongo expression inside a Python code block.
5. Do not explain — just output the code.

Example:
```python
db.movies.find({{}}, {{"title": 1, "imdb.rating": 1, "_id": 0}}).sort("imdb.rating", -1).limit(10)
```
"""
        with st.spinner("🤔 Generating query…"):
            ai_reply = ask_gemini(model, ai_prompt)

        query_code = extract_python_query(ai_reply)

        # ── Execute the generated query ──────────────────────────────────────────
        with st.spinner("⚡ Running query on MongoDB…"):
            results, error = execute_query(client, query_code)

        # ── Format assistant reply ───────────────────────────────────────────────
        with st.chat_message("assistant"):
            st.markdown("**Generated MongoDB Query:**")
            st.code(query_code, language="python")

            if error:
                st.error(f"Query error: {error}")
                reply_text = f"❌ Query error: {error}"
            elif results is None:
                st.warning("No results returned.")
                reply_text = "⚠️ No results returned."
            elif isinstance(results, list) and len(results) == 0:
                st.info("Query ran successfully but returned 0 documents.")
                reply_text = "Query ran but returned 0 documents."
            elif isinstance(results, list):
                safe_results = json_safe(results)
                st.markdown(f"**Results** ({len(safe_results)} document(s)):")
                st.dataframe(safe_results, use_container_width=True)
                reply_text = f"Returned {len(safe_results)} document(s)."
            else:
                st.write(results)
                reply_text = str(results)

        st.session_state.history.append({
            "role": "assistant",
            "content": f"Query:\n```python\n{query_code}\n```\n\n{reply_text}"
        })

    # ── Clear chat ───────────────────────────────────────────────────────────────
    if st.session_state.history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.rerun()
