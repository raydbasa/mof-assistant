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

# Load env - for local development
load_dotenv()

def get_secret(key: str) -> Optional[str]:
    """
    Get secret from Streamlit secrets (Cloud) or environment variables (local).
    Streamlit Cloud uses st.secrets, local dev uses .env
    """
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fall back to environment variables (for local development)
    return os.getenv(key)

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
    missing = [v for v in required_vars if not get_secret(v)]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.info("💡 For Streamlit Cloud: Add secrets in Settings > Secrets. For local: create .env file.")
        return False
    return True

def sanitize_input(text: str) -> str:
    """Basic sanitization for logging (mask keys, limit length)."""
    if not text:
        return ""
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY]', text)
    text = re.sub(r'pk-[a-zA-Z0-9]{20,}', '[PUBLIC_KEY]', text)
    return text[:500]

def validate_text_input(text: str, max_length: int = 10000, field_name: str = "Input") -> bool:
    """
    Validate text input for security and reasonable limits.
    Returns True if valid, False otherwise (with error message displayed).
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check length
    if len(text) > max_length:
        st.error(f"❌ {field_name} is too long. Maximum {max_length} characters allowed.")
        return False
    
    # Check for potential injection attempts (basic protection)
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            st.error(f"❌ {field_name} contains potentially unsafe content.")
            return False
    
    return True

def check_rate_limit(key: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
    """
    Simple rate limiting using session state.
    Returns True if request is allowed, False otherwise.
    """
    import time
    
    if 'rate_limit' not in st.session_state:
        st.session_state.rate_limit = {}
    
    current_time = time.time()
    
    if key not in st.session_state.rate_limit:
        st.session_state.rate_limit[key] = []
    
    # Clean old requests outside the window
    st.session_state.rate_limit[key] = [
        timestamp for timestamp in st.session_state.rate_limit[key]
        if current_time - timestamp < window_seconds
    ]
    
    # Check if limit exceeded
    if len(st.session_state.rate_limit[key]) >= max_requests:
        st.warning(f"⏱️ Rate limit reached. Please wait {window_seconds} seconds before trying again.")
        return False
    
    # Add current request
    st.session_state.rate_limit[key].append(current_time)
    return True
