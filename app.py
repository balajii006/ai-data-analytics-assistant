import os
import re
import sqlite3
import threading

import pandas as pd
import gradio as gr
from matplotlib.figure import Figure
from google import genai
from google.genai import types

# ---------------- Gemini setup (same API key) ----------------
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),  # 30 sec timeout
)

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = "gemini-2.5-flash"


def ask_gemini(prompt):
    last_error = None
    for model in dict.fromkeys([PRIMARY_MODEL, FALLBACK_MODEL]):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            return resp.text.strip()
        except Exception as e:
            last_error = e
    raise last_error


# ---------------- CSV cache (fast) ----------------
_lock = threading.Lock()
_cache = {"key": None}


def load_data(path):
    key = (path, os.path.getmtime(path))
    with _lock:
        if _cache.get("key") == key:
            return _cache

        df = pd.read_csv(path)
        df = df.drop_duplicates().dropna(how="all")
        df.columns = df.columns.astype(str).str.strip()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql("sales", conn, if_exists="replace", index=False)
        conn.execute("PRAGMA query_only = ON")  # read-only

        schema_info = pd.read_sql("PRAGMA table_info(sales)", conn)
        schema = "\n".join(
            f'- "{r["name"]}" ({r["type"]})' for _, r in schema_info.iterrows()
        )
        samples = df.head(3).to_string(index=False)

        if _cache.get("conn") is not None:
            try:
                _cache["conn"].close()
            except Exception:
                pass

        _cache.update(
            key=key, df=df, conn=conn, schema=schema, samples=samples
        )
        return _cache


# ---------------- Dashboard summary on upload ----------------
def on_upload(file):
    if file is None:
        return "Upload a CSV to see summary.", None
    try:
        c = load_data(file)
    except Exception as e:
        return f"❌ Could not read CSV: {e}", None

    df = c["df"]
    num_cols = df.select_dtypes("number").columns.tolist()

    md = f"### 📁 Dataset Summary\n\n**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}\n\n"
    if num_cols:
        md += "| Metric | Total | Average |\n|---|---|---|\n"
        for col in num_cols[:6]:
            md += f"| {col} | {df[col].sum():,.2f} | {df[col].mean():,.2f} |\n"
    return md, df.head(10)


# ---------------- SQL safety ----------------
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.I,
)


def is_safe(query):
    if not re.match(r"^\s*(select|with)\b", query, re.I):
        return False, "Invalid SQL generated."
    # ignore text inside quotes (column names / values)
    stripped = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', "", query).rstrip("; \n")
    if ";" in stripped:
        return False, "Multiple statements not allowed."
    if FORBIDDEN.search(stripped):
        return False, "Unsafe SQL generated."
    return True, ""


# ---------------- Fast local insight ----------------
def quick_insight(result):
    if result.empty:
        return "No data found for this question."

    if result.shape[1] >= 2 and pd.api.types.is_numeric_dtype(result.iloc[:, -1]):
        label, metric = result.columns[0], result.columns[-1]
        top = result.loc[result[metric].idxmax()]
        low = result.loc[result[metric].idxmin()]
        total = result[metric].sum()
        text = f"Highest {metric}: {top[label]} ({top[metric]:,.2f})."
        if len(result) > 1:
            text += f" Lowest: {low[label]} ({low[metric]:,.2f})."
            if total:
                text += f" Top one contributes {top[metric] / total * 100:.1f}% of total."
        return text

    return f"Query returned {len(result)} row(s)."


def ai_insight(question, result):
    prompt = f"""You are a business analyst.
Question: {question}
SQL result:
{result.head(30).to_string(index=False)}

Give ONE short business insight based ONLY on this result. Do not invent information."""
    try:
        return ask_gemini(prompt)
    except Exception:
        return quick_insight(result)


# ---------------- Chart ----------------
def make_chart(result, question):
    if result.shape[1] != 2 or len(result) > 30:
        return None
    x_col, y_col = result.columns
    if not pd.api.types.is_numeric_dtype(result[y_col]):
        return None

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.bar(result[x_col].astype(str), result[y_col])
    ax.set_title(question[:80])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------- Main analyze ----------------
def analyze_data(file, question, use_ai):
    if file is None:
        return None, None, "Please upload a CSV file."
    if not question or not question.strip():
        return None, None, "Please enter a business question."

    try:
        c = load_data(file)
    except Exception as e:
        return None, None, f"Could not read CSV: {e}"

    prompt = f"""You are a SQL analyst.

Table: sales
Columns:
{c['schema']}

Sample rows:
{c['samples']}

User question:
{question}

Generate ONLY one SQLite SELECT query.

Rules:
- Use only columns that exist. Wrap every column name in double quotes.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or REPLACE.
- Return only SQL, no explanation, no markdown.
- For highest/lowest/most/least questions, return the category column AND the numeric metric, with ORDER BY and LIMIT.
- Sales by region -> Region and SUM(Sales). Profit by product -> Product and SUM(Profit).
- Always include the numeric value in SELECT for ranking questions.
"""

    try:
        query = ask_gemini(prompt)
    except Exception as e:
        return None, None, f"Gemini error: {e}"

    query = query.replace("```sql", "").replace("```", "").strip()

    ok, msg = is_safe(query)
    if not ok:
        return None, None, f"{msg}\n\nQuery: {query}"

    try:
        with _lock:
            result = pd.read_sql(query, c["conn"])
    except Exception as e:
        return None, None, f"SQL execution error: {e}\n\nQuery: {query}"

    chart = make_chart(result, question)
    insight = ai_insight(question, result) if use_ai else quick_insight(result)

    return result, chart, insight


# ---------------- Gradio UI ----------------
with gr.Blocks(title="AI Data Analytics Assistant") as app:
    gr.Markdown("# 📊 AI-Powered Data Analytics Assistant")
    gr.Markdown("Upload a CSV and ask a business question.")

    file = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")

    summary = gr.Markdown("Upload a CSV to see summary.")
    preview = gr.Dataframe(label="Data Preview (first 10 rows)")

    question = gr.Textbox(
        label="Ask your business question",
        placeholder="Example: Which region has the highest sales?",
    )
    use_ai = gr.Checkbox(label="Use AI for insight (slower)", value=False)
    analyze_button = gr.Button("🔍 Analyze", variant="primary")

    result_table = gr.Dataframe(label="SQL Result")
    chart = gr.Plot(label="Chart")
    insight = gr.Textbox(label="💡 Business Insight", lines=3)

    file.change(on_upload, inputs=file, outputs=[summary, preview])

    analyze_button.click(
        fn=analyze_data,
        inputs=[file, question, use_ai],
        outputs=[result_table, chart, insight],
    )

app.queue()
app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
)
