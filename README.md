[README.md](https://github.com/user-attachments/files/29800432/README.md)
# Project 1 — AI-Powered MongoDB Query System

## What it does
A Streamlit web app where you type questions in plain English and the app:
1. Reads your MongoDB collection schemas (RAG context)
2. Passes your question + schema to Google Gemini
3. Gemini generates the correct PyMongo query
4. The app runs the query and shows results in a table

---

## One-time Setup

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Get a FREE MongoDB Atlas account
1. Go to https://www.mongodb.com/cloud/atlas
2. Sign up for free (M0 free tier)
3. Create a cluster → Load **Sample Dataset** (includes `sample_mflix`)
4. Under **Database Access** → Add a user with read/write permissions
5. Under **Network Access** → Add `0.0.0.0/0` (allow all IPs for local testing)
6. Click **Connect** → **Drivers** → copy the URI  
   It looks like: `mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/`

### 3. Get a FREE Google Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API Key**
3. Copy the key (starts with `AIza`)

---

## Run the App
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser (Opera is fine).

---

## How to Demo

### Step 1 — Paste credentials in the sidebar
- Enter your MongoDB URI and Gemini API key on the left panel.

### Step 2 — Try these sample prompts on the **Process1** tab

| Prompt | Expected behaviour |
|--------|--------------------|
| Show top 10 movies with highest IMDb rating | Queries `movies` collection, sorts by `imdb.rating` |
| How many movies were released after 2010? | Uses `count_documents` with a date filter |
| List all movies in the Horror genre | Filters by `genres` field |
| Who directed the movie Inception? | Returns director info |
| Show movies with more than 1000 votes | Filters on `imdb.votes` |

### Step 3 — Show follow-up conversation
After asking about top-rated movies, follow up with:  
> "Show only the ones released after 2000"

The app remembers the last 6 turns of conversation.

---

## Architecture (for mentor questions)

```
User Prompt
    │
    ▼
Streamlit UI (Process1 tab)
    │
    ▼
RAG Context Builder ──► MongoDB Atlas (reads collection schemas)
    │
    ▼
Google Gemini 1.5 Flash (generates PyMongo query)
    │
    ▼
Query Executor (safe: read-only, capped at 50 rows)
    │
    ▼
Results displayed as DataFrame in Streamlit
```

## Key Technical Points (Mentor Q&A prep)

**Q: What is RAG in this project?**  
A: RAG (Retrieval-Augmented Generation) means we retrieve the MongoDB schema/field names and pass them as context to Gemini. This way the AI "knows" the structure of our data before generating a query.

**Q: How is it "Agentic"?**  
A: The AI decides which collection to query, what filters to apply, and how to sort/limit — all autonomously based on the user's natural language question.

**Q: Why limit results to 50?**  
A: To avoid overloading the UI and MongoDB connection with massive result sets.

**Q: What stops the AI from running destructive queries?**  
A: The executor checks that only `find`, `aggregate`, `count_documents`, or `distinct` operations are present before running any code.
