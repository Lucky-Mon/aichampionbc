
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_file(uploaded_file) -> str:
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
        return uploaded_file.read().decode("utf-8")
    else:
        return "Unsupported file type"

def parse_excel(file):
    df_raw = pd.read_excel(file, header=None)

    # Attempt to detect the correct header row
    for i in range(3):  # Adjust range if your headers start later
        if df_raw.iloc[i].notna().sum() > 2:
            df = pd.read_excel(file, header=i)
            break
    else:
        df = df_raw

    # Clean column names
    df.columns = df.columns.map(lambda x: str(x).strip().replace('\n', ' ').replace('\r', ' '))
    df = df.dropna(how="all")  # Drop fully empty rows

    insights = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        for col in numeric_cols:
            try:
                insights.append(f"**{col}**: mean = {df[col].mean():.2f}, min = {df[col].min()}, max = {df[col].max()}")
            except Exception:
                continue
    else:
        insights.append("No numeric columns found.")

    # Clean preview rows
    try:
        preview_rows = df.head(3).to_markdown(index=False)
    except Exception:
        preview_rows = df.head(3).to_string(index=False)

    return df, "\n\n".join(insights + ["\nTop 3 rows:\n", preview_rows])

def detect_document_type(text: str) -> str:
    '''
    Improved heuristic for detecting document type:
    - If majority of content is narrative, return 'informational'
    - If large portion is numeric or tabular, return 'data'
    '''
    lines = text.splitlines()
    num_lines = len(lines)

    # Heuristics
    num_table_lines = sum(1 for line in lines if "," in line or "\t" in line or "|" in line)
    num_numeric_lines = sum(1 for line in lines if sum(char.isdigit() for char in line) > len(line) * 0.3)
    avg_line_length = sum(len(line) for line in lines if line.strip()) / max(1, len([l for l in lines if l.strip()]))
    word_count = len(text.split())

    if word_count < 100 or avg_line_length < 40:
        return "data"
    if (num_table_lines + num_numeric_lines) > num_lines * 0.4:
        return "data"
    
    return "informational"

def summarize_text(text: str, max_bullets=3) -> list:
    from openai import OpenAI
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Summarize the following document into {max_bullets} concise bullet points:

    {text[:3000]}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    bullets = response.choices[0].message.content
    return bullets
