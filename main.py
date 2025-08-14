# rag_qa_streamlit_app/main.py
import streamlit as st
try:
    from auth import check_login
    from rag_pipeline import initialize_vectorstore, add_document_to_store, answer_query
    from utils import parse_file, parse_excel, detect_document_type, summarize_text
except ImportError as e:
    import streamlit as st
    st.error(f"Import error: {e}")
    st.stop()
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.expander(    
    """
IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.
"""
    , expanded=False)
def login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
            else:
                st.error("Invalid credentials")
        st.stop()

# Call the login function
login()

# Main tabbed navigation
st.title("🔍 Intelligent Document Assistant")
tab1, tab2, tab3 = st.tabs(["📁 Upload & Ask", "👤 About Us", "📘 Methodology"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt", 'xlsx', 'xls'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".pdf") or uploaded_file.name.endswith(".txt"):
                text = parse_file(uploaded_file)
                doc_type = detect_document_type(text)

                if doc_type == "informational":
                    st.subheader("🧠 Informational Document Detected")
                    st.markdown("**Default Summary:**")
                    summary = summarize_text(text)
                    st.markdown(summary)

                    if st.button("🔍 Show deeper summary or more details"):
                        summary = summarize_text(text, max_bullets=7)
                        st.markdown("**🔍 Extended Summary:**")
                        st.markdown(summary)

                    vs = initialize_vectorstore()
                    add_document_to_store(vs, text)

                    with st.expander("💬 Ask a Question"):
                        query = st.text_input("Ask a question based on the document")
                        if query:
                            response = answer_query(vs, query)
                            st.write("**Answer:**", response)
                else:
                    st.subheader("📊 Structured Data Detected")
                    st.write("The system detected tabular or structured content. Please consider uploading this as an Excel file for better insights.")
            elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
                df, insights = parse_excel(uploaded_file)
                st.subheader("📊 Excel Data Insights")
                st.write(insights)

                # Display the DataFrame
                st.dataframe(df)

                # Visualize numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    st.subheader("📈 Numeric Column Visualizations")
                    for col in numeric_cols:
                        st.bar_chart(df[col])
                    vs = initialize_vectorstore()
                    add_document_to_store(vs, insights)
                    with st.expander("💬 Ask a Question"):
                        query_excel = st.text_input("Ask a question based on the document")
                        if query_excel:
                            response = answer_query(vs, query_excel)
                            st.write("**Answer:**", response)
                else:
                    st.warning("No numeric columns found for visualization.")
                
                    

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

with tab2:
    st.header("👤 About Us")
    st.markdown("""
    Welcome to the Intelligent Document Assistant — a project built for exploring the power of AI and document intelligence.

    🚀 **What it does:**
    - Uses **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded documents.
    - Automatically detects document type (structured vs. narrative).
    - Generates summaries, visual insights, and lets you query the content intelligently.

    👨‍💻 **Built by:** A solo developer passionate about applying AI to solve real-world problems in automation, business intelligence, and education.

    📦 **Tech Stack:**
    - **Streamlit** for interactive UI
    - **OpenAI GPT-3.5/4** for language understanding and summarization
    - **FAISS/Chroma** for semantic search
    - **Pandas + Matplotlib** for data parsing and visualization
    """)

with tab3:
    st.header("📘 Methodology")
    st.markdown("""
    ## How It Works

    This app combines **AI language models** with **information retrieval** techniques to process both textual and structured documents:

    ### 📄 Informational Documents (PDF, TXT)
    - Text is extracted from files
    - Document type is auto-detected using custom heuristics
    - OpenAI GPT is used to summarize and answer questions based on retrieved content

    ### 📊 Structured Documents (Excel)
    - Excel files are parsed using `pandas`
    - Numeric columns are profiled for insights (mean, min, max)
    - Key metrics are visualized via bar charts

    ### 🧠 RAG (Retrieval-Augmented Generation)
    - Uploaded text is broken into chunks and embedded using OpenAI Embedding
    - Embeddings are stored in FAISS or Chroma for fast retrieval
    - Relevant chunks are retrieved based on query similarity
    - GPT-3.5/GPT-4 generates the final answer using both context and knowledge

    ### 🧰 Tools Used
    - **Streamlit** for user interaction
    - **OpenAI** for summarization and LLM querying
    - **FAISS/Chroma** for vector search
    - **LangChain** to orchestrate RAG
    - **dotenv + modular Python code** for clean structure and easy maintenance

    > This app is designed to be minimal yet powerful, demonstrating a complete AI-powered document Q&A pipeline.
    """)

