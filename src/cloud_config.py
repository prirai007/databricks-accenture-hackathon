"""Cloud deployment config — reads from Streamlit secrets or .env file."""

import os

try:
    import streamlit as st
    # Streamlit Cloud secrets injection
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass  # local .env is already loaded by python-dotenv