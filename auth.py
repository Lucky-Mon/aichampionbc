# rag_qa_streamlit_app/auth.py

import os
from dotenv import load_dotenv

load_dotenv()

def check_login(username: str, password: str) -> bool:
    valid_username = os.getenv("APP_USERNAME", "admin")
    valid_password = os.getenv("APP_PASSWORD", "password123")
    return username == valid_username and password == valid_password
