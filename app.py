import html
import os
import re
import sqlite3
import threading

import pandas as pd
import gradio as gr
from matplotlib.figure import Figure
from google import genai
from google.genai import types

# =====================================================
#  Gemini setup (same API key, same env variable)
# =====================================================
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),  # 30 sec timeout
)

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = "gemini-2.5-flash"

TEAL = "#0f9d8a"
AMBER = "#f5a623"
SLATE = "#94a3b8"


def ask_gemini(prompt):
    last_error = None
    for model in dict.fromkeys([PRIMARY_MODEL, FALLBACK_MODEL]):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            return resp.text.strip()
        except Exception as e:
            last_error = e
    raise last_error


# =====================================================
#  Helpers
# =====================================================
def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def pretty(col):
    return re.sub(r"\s+", " ", re.sub(r'["_]', " ", str(col))).strip().title()


# =====================================================
#  CSV cache (read once, reuse for every question)
# =====================================================
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

        _cache.update(key=key, df=df, conn=conn, schema=schema, samples=samples)
        return _cache


# =====================================================
#  Chart (highlights the top bar, shows values)
# =====================================================
def make_chart(result, title):
    if result is None or result.empty or result.shape[1] != 2 or len(result) > 30:
        return None
    x_col, y_col = result.columns
    if not pd.api.types.is_numeric_dtype(result[y_col]):
        return None

    labels = result[x_col].astype(str).tolist()
    vals = result[y_col].fillna(0).tolist()
    top_i = vals.index(max(vals))
    colors = [AMBER if i == top_i else TEAL for i in range(len(vals))]
    horizontal = len(labels) > 8 or max(len(s) for s in labels) > 12

    fig = Figure(figsize=(8, 5 if not horizontal else max(4, 0.42 * len(labels) + 1.5)))
    ax = fig.subplots()

    if horizontal:
        bars = ax.barh(labels, vals, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(pretty(y_col))
        ax.xaxis.grid(True, color="#e2e8f0")
    else:
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_ylabel(pretty(y_col))
        ax.set_xlabel(pretty(x_col))
        ax.yaxis.grid(True, color="#e2e8f0")
        ax.tick_params(axis="x", rotation=30)

    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[fmt(v) for v in vals], padding=3, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.margins(y=0.12) if not horizontal else ax.margins(x=0.15)
    ax.set_title(title[:80], fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    return fig


# =====================================================
#  Dashboard (shows right after upload)
# =====================================================
PREFERRED = ("sales", "revenue", "profit", "amount", "total", "price")


def pick_metric(df):
    nums = df.select_dtypes("number").columns.tolist()
    nums = [c for c in nums if not re.search(r"(^id$|_id$|id$)", c, re.I)] or nums
    nums.sort(key=lambda c: 0 if any(k in c.lower() for k in PREFERRED) else 1)
    return nums


def pick_category(df):
    cats = [
        c
        for c in df.select_dtypes(exclude="number").columns
        if 2 <= df[c].nunique() <= 12
    ]
    cats.sort(key=lambda c: df[c].nunique())
    return cats


def kpi_html(df):
    cards = [
        ("Rows", f"{len(df):,}"),
        ("Columns", f"{df.shape[1]}"),
    ]
    for col in pick_metric(df)[:3]:
        cards.append((f"Total {pretty(col)}", fmt(float(df[col].sum()))))
    body = "".join(
        f'<div class="kpi"><div class="kpi-label">{html.escape(l)}</div>'
        f'<div class="kpi-value">{html.escape(v)}</div></div>'
        for l, v in cards
    )
    return f'<div class="kpi-grid">{body}</div>'


EMPTY_DASH = (
    '<div class="empty">📂 Upload a CSV file. Dashboard summary and a chart '
    "will appear here automatically.</div>"
)


def on_upload(file):
    if file is None:
        return EMPTY_DASH, None, None
    try:
        c = load_data(file)
    except Exception as e:
        return f'<div class="empty error">❌ CSV read panna mudiyala: {html.escape(str(e))}</div>', None, None

    df = c["df"]
    chart = None
    metrics, cats = pick_metric(df), pick_category(df)
    if metrics and cats:
        m, g = metrics[0], cats[0]
        agg = (
            df.groupby(g)[m].sum().sort_values(ascending=False).reset_index()
        )
        chart = make_chart(agg, f"{pretty(m)} by {pretty(g)}")
    return kpi_html(df), chart, df.head(10)


# =====================================================
#  SQL safety
# =====================================================
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.I,
)


def is_safe(query):
    if not re.match(r"^\s*(select|with)\b", query, re.I):
        return False, "Invalid SQL generated."
    stripped = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', "", query).rstrip("; \n")
    if ";" in stripped:
        return False, "Multiple statements not allowed."
    if FORBIDDEN.search(stripped):
        return False, "Unsafe SQL generated."
    return True, ""


# =====================================================
#  Insight (local = instant, AI = optional)
# =====================================================
def quick_insight(result):
    if result.empty:
        return "No data found for this question."

    if result.shape[1] >= 2 and pd.api.types.is_numeric_dtype(result.iloc[:, -1]):
        label, metric = result.columns[0], result.columns[-1]
        r = result.dropna(subset=[metric])
        if r.empty:
            return f"Query returned {len(result)} row(s)."
        top = r.loc[r[metric].idxmax()]
        low = r.loc[r[metric].idxmin()]
        total = r[metric].sum()
        text = f"Highest {pretty(metric)}: {top[label]} ({fmt(top[metric])})."
        if len(r) > 1:
            text += f" Lowest: {low[label]} ({fmt(low[metric])})."
            if total:
                text += f" {top[label]} contributes {top[metric] / total * 100:.1f}% of the total."
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


def answer_html(result, insight_text):
    hero = ""
    if (
        not result.empty
        and result.shape[1] >= 2
        and pd.api.types.is_numeric_dtype(result.iloc[:, -1])
    ):
        r = result.dropna(subset=[result.columns[-1]])
        if not r.empty:
            top = r.loc[r[result.columns[-1]].idxmax()]
            hero = (
                '<div class="hero">'
                f'<div class="hero-tag">🏆 Top result</div>'
                f'<div class="hero-name">{html.escape(str(top[result.columns[0]]))}</div>'
                f'<div class="hero-value">{html.escape(fmt(top[result.columns[-1]]))} '
                f'<span>{html.escape(pretty(result.columns[-1]))}</span></div>'
                "</div>"
            )
    return (
        f'{hero}<div class="insight">💡 {html.escape(insight_text)}</div>'
    )


def error_html(msg):
    return f'<div class="empty error">⚠️ {html.escape(msg)}</div>'


# =====================================================
#  Main analyze
# =====================================================
def analyze_data(file, question, use_ai):
    if file is None:
        return None, None, error_html("Please upload a CSV file first."), ""
    if not question or not question.strip():
        return None, None, error_html("Please enter a business question."), ""

    try:
        c = load_data(file)
    except Exception as e:
        return None, None, error_html(f"Could not read CSV: {e}"), ""

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
- For highest/lowest/most/least questions, return the category column AND the numeric metric, ordered by the metric (DESC for highest, ASC for lowest). Do NOT use LIMIT unless the user asks for "top N" or "bottom N".
- Always give aggregates a clean snake_case alias, for example SUM("Sales") AS total_sales, SUM("Profit") AS total_profit.
- Sales by region -> "Region" and SUM("Sales") AS total_sales. Profit by product -> "Product" and SUM("Profit") AS total_profit.
- Always include the numeric value in SELECT for ranking questions.
"""

    try:
        query = ask_gemini(prompt)
    except Exception as e:
        return None, None, error_html(f"Gemini error: {e}"), ""

    query = query.replace("```sql", "").replace("```", "").strip()

    ok, msg = is_safe(query)
    if not ok:
        return None, None, error_html(msg), query

    try:
        with _lock:
            result = pd.read_sql(query, c["conn"])
    except Exception as e:
        return None, None, error_html(f"SQL execution error: {e}"), query

    chart = make_chart(result, question)
    insight_text = ai_insight(question, result) if use_ai else quick_insight(result)

    return result, chart, answer_html(result, insight_text), query


# =====================================================
#  UI
# =====================================================
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.banner {
  background: linear-gradient(120deg, #0f766e 0%, #0f9d8a 55%, #14b8a6 100%);
  color: #fff; border-radius: 16px; padding: 26px 30px; margin-bottom: 8px;
}
.banner h1 { margin: 0 0 6px; font-size: 1.9rem; font-weight: 700; color: #fff; }
.banner p  { margin: 0; opacity: .92; font-size: 1rem; color: #fff; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.kpi { border: 1px solid rgba(148,163,184,.35); border-left: 5px solid #0f9d8a;
       border-radius: 12px; padding: 14px 16px; background: rgba(15,157,138,.07); }
.kpi-label { font-size: .8rem; opacity: .75; margin-bottom: 4px; }
.kpi-value { font-size: 1.45rem; font-weight: 700; }
.hero { border-radius: 14px; padding: 18px 22px; margin-bottom: 10px;
        background: linear-gradient(120deg, rgba(245,166,35,.22), rgba(245,166,35,.06));
        border: 1px solid rgba(245,166,35,.55); }
.hero-tag { font-size: .85rem; opacity: .8; }
.hero-name { font-size: 1.8rem; font-weight: 800; margin: 2px 0; }
.hero-value { font-size: 1.15rem; font-weight: 600; }
.hero-value span { font-weight: 400; opacity: .7; font-size: .9rem; }
.insight { border-radius: 12px; padding: 14px 18px; line-height: 1.55;
           border: 1px solid rgba(15,157,138,.45); background: rgba(15,157,138,.08); }
.empty { border: 2px dashed rgba(148,163,184,.6); border-radius: 12px;
         padding: 22px; text-align: center; opacity: .85; }
.empty.error { border-color: rgba(239,68,68,.7); background: rgba(239,68,68,.07); opacity: 1; }
#analyze-btn { font-size: 1.05rem; }
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
)

with gr.Blocks(title="AI Data Analytics Assistant", theme=theme, css=CSS) as app:
    gr.HTML(
        '<div class="banner"><h1>📊 AI Data Analytics Assistant</h1>'
        "<p>Upload a CSV, ask in plain English, get SQL, chart and insight instantly.</p></div>"
    )

    file = gr.File(label="1. Upload your CSV", file_types=[".csv"], type="filepath")

    dash = gr.HTML(EMPTY_DASH)
    dash_chart = gr.Plot(label="Auto Dashboard Chart")
    with gr.Accordion("👀 Data preview (first 10 rows)", open=False):
        preview = gr.Dataframe(interactive=False)

    gr.Markdown("### 2. Ask your business question")
    question = gr.Textbox(
        label="Your question",
        placeholder="Example: Which region has the highest sales?",
        lines=1,
    )
    gr.Examples(
        examples=[
            ["Which region has the highest sales?"],
            ["Show total sales by region"],
            ["Top 5 products by profit"],
            ["Which category has the lowest profit?"],
        ],
        inputs=question,
        label="Try these",
    )
    with gr.Row():
        use_ai = gr.Checkbox(label="Use AI for insight (slower)", value=False)
        analyze_button = gr.Button("🔍 Analyze", variant="primary", elem_id="analyze-btn")

    gr.Markdown("### 3. Results")
    answer = gr.HTML()
    chart = gr.Plot(label="Chart")
    result_table = gr.Dataframe(label="SQL Result", interactive=False)
    with gr.Accordion("🧾 Generated SQL", open=False):
        sql_box = gr.Code(language="sql", label=None)

    file.change(on_upload, inputs=file, outputs=[dash, dash_chart, preview])

    outputs = [result_table, chart, answer, sql_box]
    analyze_button.click(analyze_data, inputs=[file, question, use_ai], outputs=outputs)
    question.submit(analyze_data, inputs=[file, question, use_ai], outputs=outputs)

app.queue()
app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
)
