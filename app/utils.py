"""
Utility functions for MOF synthesis assistant (Langfuse-free).
"""
import json
import pandas as pd
from typing import Dict, Any, Optional, List
import streamlit as st
import os
from dotenv import load_dotenv
import re

# Load env
load_dotenv()

def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from OpenAI API."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}")
        return None

def display_mof_row(row: Dict[str, Any]) -> None:
    """Display a MOF row in Streamlit."""
    if not row:
        st.error("No data to display")
        return
    df = pd.DataFrame([row])
    st.dataframe(df, use_container_width=True)

def download_csv(data: List[Dict[str, Any]], filename: str = "mof_data.csv") -> bytes:
    """Convert data to CSV for download."""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

def download_json(data: List[Dict[str, Any]], filename: str = "mof_data.json") -> str:
    """Convert data to JSON for download."""
    return json.dumps(data, indent=2)

def extract_numerical_value(text: str) -> Optional[float]:
    """Extract the first numerical value from text."""
    if not text or text.strip() == "":
        return None
    try:
        return float(text.strip())
    except ValueError:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        return None

def validate_environment() -> bool:
    """Validate required env vars."""
    required_vars = ["OPENAI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def sanitize_input(text: str) -> str:
    """Basic sanitization for logging (mask keys, limit length)."""
    if not text:
        return ""
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'pk-[a-zA-Z0-9]{20,}', '[PUBLIC_KEY]', text)
    return text[:500]
