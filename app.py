import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def analyze_data(file, question):
    if file is None:
        return None, None, "Please upload a CSV file."

    if not question or not question.strip():
        return None, None, "Please enter a business question."

    data = pd.read_csv(file)
    data = data.drop_duplicates()
    data = data.dropna(how="all")
    data.columns = data.columns.str.strip()

    connection = sqlite3.connect(":memory:")
    data.to_sql("sales", connection, if_exists="replace", index=False)

    schema_info = pd.read_sql("PRAGMA table_info(sales)", connection)
    schema = "\n".join(
        f"- {row['name']} ({row['type']})"
        for _, row in schema_info.iterrows()
    )

    prompt = f"""
You are a SQL analyst.

Database table:
sales

Columns:
{schema}

User question:
{question}

Generate ONLY one SQLite SELECT query.

Rules:
- Use only columns that exist in the schema.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or REPLACE.
- Return only SQL.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

    query = response.text.strip()
    query = query.replace("```sql", "").replace("```", "").strip()

    forbidden = [
        "insert", "update", "delete",
        "drop", "alter", "create", "replace"
    ]

    query_lower = query.lower()

    if not query_lower.startswith("select"):
        connection.close()
        return None, None, "Invalid SQL generated."

    if any(word in query_lower for word in forbidden):
        connection.close()
        return None, None, "Unsafe SQL generated."

    try:
        result = pd.read_sql(query, connection)
    except Exception as e:
        connection.close()
        return None, None, f"SQL execution error: {e}"

    chart = None

    if result.shape[1] == 2:
        x_column = result.columns[0]
        y_column = result.columns[1]

        if pd.api.types.is_numeric_dtype(result[y_column]):
            fig, ax = plt.subplots(figsize=(8, 5))
            result.plot(
                x=x_column,
                y=y_column,
                kind="bar",
                ax=ax
            )
            ax.set_title(question)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart = fig

    result_text = result.to_string(index=False)

    insight_prompt = f"""
You are a business analyst.

User question:
{question}

SQL result:
{result_text}

Give one short business insight based ONLY on this result.
Do not invent information.
"""

insight_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=insight_prompt
)

    insight = insight_response.text

    connection.close()
    return result, chart, insight


with gr.Blocks(title="AI Data Analytics Assistant") as app:
    gr.Markdown("# 📊 AI-Powered Data Analytics Assistant")
    gr.Markdown("Upload a CSV and ask a business question.")

    file = gr.File(
        label="Upload CSV",
        file_types=[".csv"],
        type="filepath"
    )

    question = gr.Textbox(
        label="Ask your business question",
        placeholder="Example: Which region has the highest sales?"
    )

    analyze_button = gr.Button("🔍 Analyze")

    result_table = gr.Dataframe(label="SQL Result")
    chart = gr.Plot(label="Chart")
    insight = gr.Textbox(label="💡 Business Insight", lines=3)

    analyze_button.click(
        fn=analyze_data,
        inputs=[file, question],
        outputs=[result_table, chart, insight]
    )

import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
