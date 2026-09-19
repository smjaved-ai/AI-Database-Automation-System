import streamlit as st
import pymongo
from groq import Groq
import json
import re
from bson import ObjectId
import datetime
import pymysql
import pandas as pd

# ─── Hardcoded Credentials ───────────────────────────────────────────────────────
MONGO_URI     = "Your Mongo URI"
GROQ_API_KEY  = "Your API Key"
TIDB_HOST     = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"
TIDB_PORT     = 4000
TIDB_USER     = "TIDB User"
TIDB_PASSWORD = "Your Pwd"
TIDB_DATABASE = "DB name"

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
    st.markdown("**Project — AI Database Automation System**")
    st.markdown("Ask questions in plain English and the AI generates and runs the MongoDB query for you.")
    st.markdown("---")
    st.markdown("**Database:** MongoDB Atlas `sample_mflix`")
    st.markdown("**AI Model:** LLaMA 3.3 70B Versatile (via Groq API)")

mongo_uri = MONGO_URI
groq_key  = GROQ_API_KEY

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
    client.admin.command("ping")
    return client


@st.cache_resource(show_spinner="Connecting to TiDB…")
def get_tidb_connection():
    conn = pymysql.connect(
        host=TIDB_HOST,
        port=TIDB_PORT,
        user=TIDB_USER,
        password=TIDB_PASSWORD,
        database=TIDB_DATABASE,
        ssl={"ca": None},
        ssl_verify_cert=False,
        ssl_verify_identity=False
    )
    return conn


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


def ask_groq(model, prompt: str) -> str:
    response = model.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


def extract_python_query(text: str) -> str:
    """Pull out the Python/PyMongo expression from the AI reply."""
    fenced = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if fenced:
        return fenced.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("db.") or line.startswith("collection"):
            return line
    return text.strip()


def extract_sql_block(text: str) -> str:
    """Pull out SQL from the AI reply."""
    fenced = re.search(r"```(?:sql)?\s*([\s\S]+?)```", text)
    if fenced:
        return fenced.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CREATE") or line.upper().startswith("SELECT"):
            return line
    return text.strip()


def execute_query(client, query_code: str, db_name="sample_mflix"):
    """Safely execute a generated PyMongo query. Only read operations allowed."""
    db = client[db_name]
    allowed_ops = ("find", "aggregate", "count_documents", "distinct")
    if not any(op in query_code for op in allowed_ops):
        return None, "Only read operations (find / aggregate / count_documents) are permitted."
    local_ns = {"db": db}
    try:
        exec(f"_result = {query_code}", local_ns)
        result = local_ns.get("_result")
        if hasattr(result, "__iter__") and not isinstance(result, (dict, str, int, float)):
            return list(result)[:50], None
        return result, None
    except Exception as exc:
        return None, str(exc)


def flatten_doc(doc: dict, prefix="") -> dict:
    """Flatten nested MongoDB document into a flat dict."""
    flat = {}
    for k, v in doc.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_doc(v, key))
        elif isinstance(v, list):
            flat[key] = ", ".join(str(i) for i in v[:5])
        else:
            flat[key] = v
    return flat


# ─── Initialise conversation memory ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Process1", "Process2"])

# ════════════════════════════════════════════════════════════════════════════════
# PROCESS 1 — Natural Language MongoDB Query
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("🤖 AI-Powered MongoDB Query System")
    st.caption("Ask questions in plain English — the AI generates and runs the MongoDB query for you.")

    try:
        client = get_mongo_client(mongo_uri)
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()

    try:
        model = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"❌ Groq initialisation failed: {e}")
        st.stop()

    st.success("✅ Connected to MongoDB Atlas  |  Groq ready")

    with st.spinner("📚 Loading schema context (RAG)…"):
        schema_ctx = build_schema_context(client)

    st.markdown("---")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a question about movies…  e.g. Show top 10 movies by IMDb rating")

    if user_prompt:
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.history[-6:]
        )

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
            ai_reply = ask_groq(model, ai_prompt)

        query_code = extract_python_query(ai_reply)

        with st.spinner("⚡ Running query on MongoDB…"):
            results, error = execute_query(client, query_code)

        with st.chat_message("assistant"):
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
            "content": reply_text
        })

    if st.session_state.history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PROCESS 2 — Agentic MongoDB → TiDB (MySQL) Data Migration
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("🔄 Agentic MongoDB → MySQL Data Migration")
    st.caption("Describe what data to migrate — the AI agent handles extraction, schema creation, and insertion.")

    try:
        client2 = get_mongo_client(mongo_uri)
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()

    try:
        model2 = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"❌ Groq initialisation failed: {e}")
        st.stop()

    try:
        tidb_conn = get_tidb_connection()
    except Exception as e:
        st.error(f"❌ TiDB connection failed: {e}")
        st.stop()

    st.success("✅ Connected to MongoDB Atlas  |  TiDB (MySQL) ready  |  Groq ready")

    st.markdown("---")

    migration_prompt = st.text_area(
        "Enter migration prompt:",
        placeholder="e.g. Extract movie title, year, genres, runtime and IMDb rating from MongoDB and store it in MySQL",
        height=80
    )

    if st.button("▶️ Run Migration", type="primary"):
        if not migration_prompt.strip():
            st.warning("Please enter a migration prompt.")
        else:
            # ── Step 1: AI identifies collection and fields ───────────────────────
            with st.spinner("🤖 Agent analysing prompt…"):
                schema_ctx_p2 = build_schema_context(client2)
                step1_prompt = f"""
You are a data migration agent. Based on the user prompt, identify:
1. The MongoDB collection name
2. The list of fields to extract (use dot notation for nested fields e.g. imdb.rating)

SCHEMA CONTEXT:
{schema_ctx_p2}

USER PROMPT: {migration_prompt}

Respond ONLY in this JSON format with no explanation:
{{"collection": "movies", "fields": ["title", "year", "genres", "runtime", "imdb.rating"]}}
"""
                step1_reply = ask_groq(model2, step1_prompt)
                json_match = re.search(r"\{[\s\S]+\}", step1_reply)
                if not json_match:
                    st.error("❌ Agent could not parse collection and fields.")
                    st.stop()
                migration_info = json.loads(json_match.group())
                collection_name = migration_info["collection"]
                fields = migration_info["fields"]

            # Build unique table name from collection + selected fields
            import re as _re
            fields_suffix = "_".join(
                _re.sub(r"[^a-zA-Z0-9]", "", f.split(".")[-1])[:6]
                for f in fields[:4]
            ).lower()
            table_name = f"{collection_name}_{fields_suffix}"[:60]

            st.info(f"📦 Collection: `{collection_name}` | Fields: `{', '.join(fields)}` | Table: `{table_name}`")

            # ── Step 2: Fetch data from MongoDB ──────────────────────────────────
            with st.spinner("📥 Fetching data from MongoDB…"):
                projection = {"_id": 0}
                for f in fields:
                    projection[f] = 1
                db2 = client2["sample_mflix"]
                raw_docs = list(db2[collection_name].find({}, projection).limit(100))

            st.info(f"✅ Fetched {len(raw_docs)} documents from MongoDB")

            # ── Step 3: Flatten documents ─────────────────────────────────────────
            with st.spinner("🔧 Flattening nested fields…"):
                flat_docs = [flatten_doc(doc) for doc in raw_docs]
                df = pd.DataFrame(flat_docs)
                df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
                df = df.where(pd.notnull(df), None)

            # ── Step 4: AI generates MySQL CREATE TABLE ───────────────────────────
            with st.spinner("🤖 Agent generating MySQL schema…"):
                sample_row = df.iloc[0].to_dict() if len(df) > 0 else {}
                step4_prompt = f"""
You are a MySQL schema expert. Generate a CREATE TABLE statement for TiDB (MySQL compatible).

Table name: {table_name}
Columns and sample values: {json.dumps({k: str(v)[:50] for k, v in sample_row.items()}, default=str)}

Rules:
- Use VARCHAR(500) for text fields
- Use INT for integer numbers
- Use FLOAT for decimal numbers
- Always add id INT AUTO_INCREMENT PRIMARY KEY as the first column
- Keep it simple, no foreign keys
- Return ONLY the CREATE TABLE SQL, no explanation

```sql
CREATE TABLE ...
```
"""
                step4_reply = ask_groq(model2, step4_prompt)
                create_sql = extract_sql_block(step4_reply)
                # Drop old table with same name and recreate fresh
                create_sql = create_sql.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE", 1)
                create_sql = create_sql.replace("CREATE TABLE", "CREATE TABLE", 1)

            # ── Step 5: Drop + Create table in TiDB ───────────────────────────────
            with st.spinner("🛠️ Creating table in TiDB…"):
                try:
                    cursor = tidb_conn.cursor()
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    tidb_conn.commit()
                    cursor.execute(create_sql)
                    tidb_conn.commit()
                except Exception as e:
                    st.error(f"❌ Table creation failed: {e}")
                    st.stop()

            st.info(f"✅ Table `{table_name}` ready in TiDB")

            # ── Step 6: Insert data into TiDB ─────────────────────────────────────
            with st.spinner("📤 Inserting data into TiDB…"):
                try:
                    cursor = tidb_conn.cursor()
                    cursor.execute(f"DESCRIBE {table_name}")
                    db_columns = [row[0] for row in cursor.fetchall() if row[0] != "id"]
                    insert_cols = [c for c in db_columns if c in df.columns]
                    insert_df = df[insert_cols]
                    placeholders = ", ".join(["%s"] * len(insert_cols))
                    col_names = ", ".join(insert_cols)
                    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
                    rows_inserted = 0
                    for _, row in insert_df.iterrows():
                        vals = tuple(None if pd.isna(v) else v for v in row.values)
                        try:
                            cursor.execute(insert_sql, vals)
                            rows_inserted += 1
                        except Exception:
                            continue
                    tidb_conn.commit()
                except Exception as e:
                    st.error(f"❌ Data insertion failed: {e}")
                    st.stop()

            # ── Step 7: Run SELECT and show results ───────────────────────────────
            with st.spinner("🔍 Running SELECT query…"):
                select_sql = f"SELECT * FROM {table_name} LIMIT 20"
                cursor.execute(select_sql)
                rows = cursor.fetchall()
                col_names_out = [desc[0] for desc in cursor.description]
                result_df = pd.DataFrame(rows, columns=col_names_out)

            # ── Display Output ────────────────────────────────────────────────────
            st.markdown("---")
            st.success("✅ Successfully captured data from MongoDB and stored into MySQL")
            st.markdown(f"**Rows Inserted:** {rows_inserted}")
            st.markdown(f"**SELECT Query:** `{select_sql}`")
            st.markdown("**Sample Output:**")
            st.dataframe(result_df, use_container_width=True)
