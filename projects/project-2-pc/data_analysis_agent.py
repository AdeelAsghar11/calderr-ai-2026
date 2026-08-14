import os
import io
import traceback
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Data Analysis Agent")
st.caption("CalderR Internship 2026 · Project 2-P-C")
st.markdown("Upload a CSV, ask a question in plain English — the agent writes and runs pandas code to answer it.")

# ── Sample datasets ───────────────────────────────────────────────
SAMPLE_DATA = {
    "Sales Data": pd.DataFrame({
        "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "revenue": [45000,52000,48000,61000,58000,72000,69000,74000,81000,77000,85000,92000],
        "units_sold": [120,145,132,168,155,192,185,198,214,205,228,247],
        "region": ["North","North","South","South","East","East","West","West","North","South","East","West"],
        "expenses": [32000,37000,35000,44000,41000,51000,49000,52000,57000,54000,60000,65000],
    }),
    "Employee Data": pd.DataFrame({
        "name": ["Ali","Sara","Ahmed","Fatima","Hassan","Zara","Omar","Hina","Bilal","Nadia"],
        "department": ["Engineering","Marketing","Engineering","HR","Engineering","Marketing","HR","Engineering","Marketing","HR"],
        "salary": [85000,72000,92000,68000,78000,75000,65000,88000,70000,67000],
        "experience_years": [5,3,7,4,6,4,2,8,3,5],
        "performance_score": [4.2,3.8,4.5,4.0,4.1,3.9,3.5,4.7,3.8,4.1],
    }),
    "Student Grades": pd.DataFrame({
        "student": [f"Student_{i}" for i in range(1, 21)],
        "math": [78,85,92,67,74,88,95,61,83,79,91,55,87,73,69,96,82,77,84,90],
        "science": [82,79,88,71,76,84,91,65,80,75,89,60,85,70,72,93,78,74,87,88],
        "english": [75,88,85,73,80,79,87,69,84,77,86,63,90,75,70,89,81,76,83,91],
        "grade": ["B","B","A","C","C","B","A","D","B","B","A","F","B","C","C","A","B","B","B","A"],
    }),
}

# ── LLM setup ─────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Schema analyzer ───────────────────────────────────────────────
def analyze_schema(df: pd.DataFrame) -> str:
    schema_lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "\nColumns:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        if df[col].dtype in ["int64", "float64"]:
            schema_lines.append(
                f"  - {col} ({dtype}): min={df[col].min()}, max={df[col].max()}, "
                f"mean={df[col].mean():.2f}, nulls={nulls}"
            )
        else:
            unique = df[col].nunique()
            sample = df[col].dropna().unique()[:3].tolist()
            schema_lines.append(
                f"  - {col} ({dtype}): {unique} unique values, sample={sample}, nulls={nulls}"
            )
    schema_lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(schema_lines)


# ── Code generator ────────────────────────────────────────────────
CODE_GEN_PROMPT = """You are an expert Python data analyst. Given a pandas DataFrame schema and a question, write Python code to answer it.

STRICT RULES:
1. The DataFrame is already loaded as variable `df` — do NOT reload or redefine it
2. Store your final text answer in a variable called `result_text` (must be a string)
3. If a chart would help, create matplotlib figure stored as `fig` (a plt.Figure object)
4. If no chart is needed, do NOT create or assign `fig`
5. Available: pandas as pd, numpy as np, matplotlib.pyplot as plt (all pre-imported)
6. Do NOT use: seaborn, plotly, or any other library
7. Return ONLY raw Python code — no markdown, no backticks, no explanation
8. Always handle potential errors (empty groups, missing values, division by zero)

DataFrame Schema:
{schema}

Question: {question}

Python code:"""


def generate_code(schema: str, question: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("user", CODE_GEN_PROMPT)
    ])
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"schema": schema, "question": question})
    # Strip markdown if LLM adds it anyway
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


# ── Code executor ─────────────────────────────────────────────────
def execute_code(code: str, df: pd.DataFrame) -> tuple[str, plt.Figure | None, str]:
    """
    Execute generated code in a controlled namespace.
    Returns: (result_text, fig_or_None, error_or_empty)
    """
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "result_text": "",
        "fig": None,
    }
    try:
        plt.close("all")
        exec(code, namespace)  # nosec — educational use, not production
        result_text = str(namespace.get("result_text", "No result_text set."))
        fig = namespace.get("fig", None)
        # Capture any figures created without explicit assignment
        if fig is None and plt.get_fignums():
            fig = plt.gcf()
        return result_text, fig, ""
    except Exception as e:
        error = traceback.format_exc()
        return "", None, error


# ── Narrative generator ───────────────────────────────────────────
def generate_narrative(question: str, result: str, schema: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst writing a brief report narrative. Be concise (3-4 sentences)."),
        ("user",
         f"Question: {question}\n\n"
         f"Analysis result: {result}\n\n"
         f"Dataset info: {schema[:300]}\n\n"
         "Write a brief narrative interpretation of this result.")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Source")

    data_source = st.radio("Choose data source", ["Upload CSV", "Use sample data"])

    df = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} cols")

    else:
        sample_name = st.selectbox("Select dataset", list(SAMPLE_DATA.keys()))
        df = SAMPLE_DATA[sample_name]
        st.success(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} cols")

    if df is not None:
        st.divider()
        st.header("💬 Ask a Question")
        question = st.text_area(
            "Your question",
            placeholder="e.g. Which month had the highest revenue?\nWhat is the average salary by department?",
            height=100
        )

        # Suggested questions based on sample data
        if data_source == "Use sample data":
            st.caption("💡 Try:")
            suggestions = {
                "Sales Data": [
                    "Which month had the highest revenue?",
                    "Show revenue vs expenses by month as a bar chart",
                    "What is the total revenue by region?",
                ],
                "Employee Data": [
                    "What is the average salary by department?",
                    "Show the correlation between experience and salary",
                    "Who are the top 3 performers by score?",
                ],
                "Student Grades": [
                    "What is the average score per subject?",
                    "Show grade distribution as a pie chart",
                    "Which students scored above 85 in all subjects?",
                ],
            }
            for s in suggestions.get(sample_name, []):
                if st.button(s, use_container_width=True):
                    question = s

        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────
if df is not None:

    # Always show data preview
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing values", df.isnull().sum().sum())

    if "question" in dir() and question and analyze:

        schema = analyze_schema(df)

        # ── Pipeline ──────────────────────────────────────────────
        st.divider()
        st.subheader(f"🔍 {question}")

        with st.status("Running analysis pipeline...", expanded=True) as status:

            st.write("Step 1: Analyzing dataset schema...")
            st.code(schema, language="text")

            st.write("Step 2: Generating pandas code...")
            code = generate_code(schema, question)
            st.code(code, language="python")

            st.write("Step 3: Executing code...")
            result_text, fig, error = execute_code(code, df)

            if error:
                st.error(f"Execution error:\n{error}")
                st.write("Step 4: Retrying with error context...")
                retry_prompt = f"The following code produced an error:\n\n{code}\n\nError:\n{error}\n\nFix the code. Same rules apply."
                code2 = generate_code(schema, retry_prompt)
                result_text, fig, error2 = execute_code(code2, df)
                if error2:
                    st.error("Retry also failed.")
                    status.update(label="Analysis failed", state="error")
                    st.stop()
                else:
                    st.success("✓ Retry succeeded")
                    code = code2

            st.write("Step 4: Generating narrative...")
            narrative = generate_narrative(question, result_text, schema)
            status.update(label="✓ Analysis complete", state="complete")

        # ── Results tabs ──────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Answer", "📊 Chart", "🔢 Data", "💻 Code"])

        with tab1:
            st.subheader("Answer")
            st.success(result_text)
            st.subheader("Interpretation")
            st.write(narrative)

        with tab2:
            if fig:
                st.pyplot(fig)
            else:
                st.info("No chart was generated for this question. Try asking for a chart explicitly.")

        with tab3:
            st.dataframe(df, use_container_width=True)

        with tab4:
            st.code(code, language="python")
            st.caption("This code was generated by the AI and executed on your data.")

else:
    st.info("👈 Upload a CSV or select a sample dataset to get started.")
    st.markdown("""
    **Example questions you can ask:**
    - *Which category had the highest sales?*
    - *Show monthly revenue as a line chart*
    - *What is the average salary by department?*
    - *Find all rows where profit is negative*
    - *What percentage of students passed?*
    """)
