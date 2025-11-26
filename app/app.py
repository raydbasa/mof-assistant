"""
MOF Synthesis Assistant - Streamlit Web Application (GPT-only + Quick Complete tab)
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Local modules
from schema import TABLE_SCHEMA, RESPONSE_FORMAT, validate_mof_row, clean_mof_row, COLUMN_NAMES
from prompts import SYSTEM_PROMPT, format_extraction_prompt, format_suggestion_prompt
from utils import (
    parse_json_response, display_mof_row,
    download_csv, download_json, extract_numerical_value,
    validate_environment, sanitize_input, get_secret,
    validate_text_input, check_rate_limit
)

# Constants
PRESSURE_OPTIONS = ["", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"]

# Page config
st.set_page_config(
    page_title="MOF Synthesis Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic theming and layout tweaks
st.markdown(
    """
    <style>
    /* Global background and typography */
    .main {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 35%, #020617 100%);
        color: #e5e7eb;
    }
    section.main > div {
        padding-top: 1.5rem;
    }
    .stApp {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f9fafb !important;
    }

    /* Card-like containers */
    .mof-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.9rem;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
        margin-bottom: 1.2rem;
    }

    /* Tabs */
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 0.45rem 0.9rem !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        background: rgba(15, 23, 42, 0.85) !important;
        color: #e5e7eb !important;
        font-size: 0.89rem !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #22c55e, #2dd4bf) !important;
        color: #020617 !important;
        border-color: transparent !important;
        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.45);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div { 
        background: rgba(15, 23, 42, 0.9);
        border-radius: 0.7rem;
        border: 1px solid rgba(148, 163, 184, 0.55);
        color: #e5e7eb;
    }
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label {
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 0.75rem;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.85);
        color: #e5e7eb;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        background: rgba(30, 41, 59, 0.95);
        border-color: rgba(148, 163, 184, 0.6);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-color: transparent;
        color: #020617;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #16a34a, #15803d);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(34, 197, 94, 0.5);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 0.75rem;
        padding: 0.5rem 1.2rem;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border: none;
        color: #ffffff;
        font-weight: 600;
        transition: all 0.25s;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.45);
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 0.9rem;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.45);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Success/Error/Warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 0.75rem;
        padding: 0.85rem 1.1rem;
        border-left-width: 4px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #22c55e;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617 40%, #0b1120 100%);
        border-right: 1px solid rgba(30, 64, 175, 0.7);
    }
    [data-testid="stSidebar"] h2 {
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }

    /* Subheaders */
    .stSubheader {
        color: #cbd5e1;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: rgba(148, 163, 184, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
for key in ("extracted_data", "predicted_data", "suggested_data", "completed_data"):
    if key not in st.session_state:
        st.session_state[key] = []

# ===== Helpers =====
def models_available() -> bool:
    """Return True if all ML model files exist (enables Predict tab)."""
    required = [
        'models/preproc.joblib',
        'models/xgb_temp.joblib',
        'models/xgb_time.joblib',
        'models/xgb_method.joblib',
    ]
    return all(os.path.exists(p) for p in required)

def load_ml_models() -> Dict[str, Any]:
    """Load pre-trained ML models for prediction."""
    models = {}
    model_paths = {
        'preprocessor': 'models/preproc.joblib',
        'temp_model': 'models/xgb_temp.joblib',
        'time_model': 'models/xgb_time.joblib',
        'method_model': 'models/xgb_method.joblib'
    }
    for name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load {name}: {e}")
    return models

def predict_missing_fields(input_data: Dict[str, str], models: Dict[str, Any]) -> Dict[str, Any]:
    """Predict missing fields using ML models."""
    predictions = {}
    try:
        # Build features
        features = []
        feature_names = []
        for field in ['Metal source (mmol)', 'Linker(s) (mmol)', 'Solvent(s) (mL)',
                      'Modulator / Additive', 'Pressure', 'Method']:
            if field in input_data:
                features.append(input_data[field])
                feature_names.append(field)
        if not features:
            st.warning("No features available for prediction")
            return predictions

        X = pd.DataFrame([features], columns=feature_names)

        # Use preprocessor if available
        if 'preprocessor' in models:
            try:
                X_processed = models['preprocessor'].transform(X)
            except Exception as e:
                st.warning(f"Preprocessing failed: {e}")
                X_processed = X
        else:
            X_processed = X

        # Temperature
        if 'temp_model' in models and 'Temp (°C)' not in input_data:
            try:
                temp_pred = models['temp_model'].predict(X_processed)[0]
                predictions['Temp (°C)'] = f"{temp_pred:.1f}"
            except Exception as e:
                st.warning(f"Temperature prediction failed: {e}")

        # Time
        if 'time_model' in models and 'Time (h)' not in input_data:
            try:
                time_pred = models['time_model'].predict(X_processed)[0]
                predictions['Time (h)'] = f"{time_pred:.1f}"
            except Exception as e:
                st.warning(f"Time prediction failed: {e}")

        # Method
        if 'method_model' in models and 'Method' not in input_data:
            try:
                method_pred = models['method_model'].predict(X_processed)[0]
                predictions['Method'] = str(method_pred)
            except Exception as e:
                st.warning(f"Method prediction failed: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return predictions

# ======= NEW: Quick Complete from Metal + Linker =======
def build_quick_complete_prompt(
    target: str,
    metal: str,
    linker: str,
    solvent: str,
    modulator: str,
    pressure: str,
    method_hint: str,
    notes: str
) -> List[Dict[str, str]]:
    """
    Build a concise prompt that asks the model to generate a FULL synthesis row
    given minimal cues (metal + linker, optionally target/solvent/etc.)
    Returns OpenAI chat messages list.
    """
    user_cues = {
        "Known": {
            "Target": target,
            "Metal source (mmol)": metal,
            "Linker(s) (mmol)": linker,
            "Solvent(s) (mL)": solvent,
            "Modulator / Additive": modulator,
            "Pressure": pressure,
            "Method (hint)": method_hint,
            "Notes": notes
        },
        "Goal": "Return a SINGLE valid JSON object following the project schema with ALL fields filled realistically."
    }

    system = (
        "You are a MOF synthesis assistant. "
        "Given minimal cues (often just metal precursor and linker), propose a realistic, lab-usable synthesis protocol "
        "as ONE JSON object following the exact keys of the schema. "
        "Units: Temp in °C, Time in hours, solvents in mL (simple numbers allowed), mmol for sources/modulators if applicable. "
        "Keep text concise but specific (e.g., 'DMF (10 mL)'), and avoid hallucinating exotic reagents unless necessary. "
        "Prefer common literature conditions for the given metal/linker pair (e.g., DMF, solvothermal 80–150 °C, 6–48 h) unless hints say otherwise. "
        "No explanation outside of JSON."
    )

    # We still use RESPONSE_FORMAT from schema to enforce JSON
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_cues, ensure_ascii=False)}
    ]
    return messages

def quick_complete_tab() -> None:
    st.markdown(
        "<div class='mof-card'><h2>⚗️ Complete (Metal + Linker)</h2><p>Provide metal precursor and linker (plus optional hints) and get a complete synthesis protocol.</p></div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("### 🔬 Input Parameters")

    c1, c2 = st.columns(2)
    with c1:
        target = st.text_input("Target (optional)", placeholder="e.g., MOF-5")
        metal = st.text_input("Metal source (mmol)", placeholder="e.g., Zn(NO3)2·6H2O (0.5)")
        linker = st.text_input("Linker(s) (mmol)", placeholder="e.g., H2BDC (0.5)")
        solvent = st.text_input("Solvent(s) (mL) - optional", placeholder="e.g., DMF (10)")
    with c2:
        modulator = st.text_input("Modulator / Additive - optional", placeholder="e.g., HCl (0.1)")
        pressure  = st.selectbox("Pressure (optional)", PRESSURE_OPTIONS)
        method_hint = st.text_input("Method hint (optional)", placeholder="e.g., solvothermal")
        notes = st.text_area("Notes/constraints (optional)", height=90, placeholder="e.g., prefer ~100–130 °C; time < 24 h")

    st.markdown("---")
    disabled = (metal.strip() == "" and linker.strip() == "")
    if st.button("🚀 Generate Complete Protocol", type="primary", disabled=disabled):
        # Check rate limit
        if not check_rate_limit("complete", max_requests=10, window_seconds=60):
            return
        
        if not validate_environment():
            return
        try:
            from openai import OpenAI
            client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return

        with st.spinner("Generating synthesis protocol..."):
            messages = build_quick_complete_prompt(target, metal, linker, solvent, modulator, pressure, method_hint, notes)
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=RESPONSE_FORMAT,
                    temperature=0.25
                )
                response_text = resp.choices[0].message.content
                row = parse_json_response(response_text)

                if row and validate_mof_row(row):
                    cleaned = clean_mof_row(row)
                    st.session_state.completed_data.append(cleaned)
                    st.success("✅ Generated a complete synthesis protocol!")
                    display_mof_row(cleaned)
                else:
                    st.error("❌ Model returned invalid or incomplete JSON. Try adding a bit more detail (e.g., solvent).")
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

    if st.session_state.completed_data:
        st.markdown("---")
        st.subheader("📦 Completed Protocols")
        st.markdown(f"*{len(st.session_state.completed_data)} protocol(s) generated*")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = download_csv(st.session_state.completed_data)
            st.download_button("📥 Download CSV", csv_data, "completed_mof_data.csv", "text/csv", key="complete_csv_dl", use_container_width=True)
        with c2:
            json_data = download_json(st.session_state.completed_data)
            st.download_button("📥 Download JSON", json_data, "completed_mof_data.json", "application/json", key="complete_json_dl", use_container_width=True)
        df = pd.DataFrame(st.session_state.completed_data)
        st.dataframe(df, use_container_width=True)

# ===== Tabs =====
def extract_tab() -> None:
    st.markdown(
        "<div class='mof-card'><h2>🧪 Extract MOF Synthesis Data</h2><p>Extract structured synthesis information from free text using AI.</p></div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("### 📝 Input Text")

    text_input = st.text_area(
        "Enter MOF synthesis text:",
        height=200,
        placeholder="Paste your MOF synthesis description here..."
    )

    st.markdown("---")
    if st.button("🔍 Extract Row", type="primary") and text_input.strip():
        # Validate input
        if not validate_text_input(text_input, max_length=5000, field_name="Synthesis text"):
            return
        
        # Check rate limit
        if not check_rate_limit("extract", max_requests=10, window_seconds=60):
            return
        
        if not validate_environment():
            return

        # OpenAI client
        try:
            from openai import OpenAI
            client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return

        try:
            with st.spinner("Extracting synthesis data..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_extraction_prompt(text_input)}
                ]
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=RESPONSE_FORMAT,
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
                extracted_row = parse_json_response(response_text)

                if extracted_row and validate_mof_row(extracted_row):
                    cleaned_row = clean_mof_row(extracted_row)
                    st.session_state.extracted_data.append(cleaned_row)
                    st.success("✅ Successfully extracted synthesis data!")
                    display_mof_row(cleaned_row)
                else:
                    st.error("❌ Failed to extract valid synthesis data")

        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")

    if st.session_state.extracted_data:
        st.markdown("---")
        st.subheader("📊 Extracted Data")
        st.markdown(f"*{len(st.session_state.extracted_data)} row(s) extracted*")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = download_csv(st.session_state.extracted_data)
            st.download_button("📥 Download CSV", csv_data, "extracted_mof_data.csv", "text/csv", use_container_width=True)
        with c2:
            json_data = download_json(st.session_state.extracted_data)
            st.download_button("📥 Download JSON", json_data, "extracted_mof_data.json", "application/json", use_container_width=True)

        df = pd.DataFrame(st.session_state.extracted_data)
        st.dataframe(df, use_container_width=True)

def predict_tab() -> None:
    st.markdown(
        "<div class='mof-card'><h2>🔮 Predict Missing Fields</h2><p>Predict missing synthesis parameters using local ML models.</p></div>",
        unsafe_allow_html=True,
    )

    models = load_ml_models()
    if not models:
        st.warning("⚠️ No ML models found. Please train models first.")
        return

    st.markdown("### 🧬 Input Known Parameters")
    c1, c2 = st.columns(2)
    with c1:
        target = st.text_input("Target MOF", placeholder="e.g., MOF-5")
        metal_source = st.text_input("Metal source (mmol)", placeholder="e.g., Zn(NO3)2·6H2O (0.5)")
        linker = st.text_input("Linker(s) (mmol)", placeholder="e.g., H2BDC (0.5)")
        solvent = st.text_input("Solvent(s) (mL)", placeholder="e.g., DMF (10)")
        modulator = st.text_input("Modulator / Additive", placeholder="e.g., H2O (0.5)")
    with c2:
        pressure = st.selectbox("Pressure", PRESSURE_OPTIONS)
        method = st.text_input("Method", placeholder="e.g., solvothermal")
        wash_activation = st.text_input("Wash / Activation", placeholder="e.g., DMF, methanol")
        predict_temp = st.checkbox("Predict Temperature (°C)", value=True)
        predict_time = st.checkbox("Predict Time (h)", value=True)
        predict_method = st.checkbox("Predict Method", value=False)

    st.markdown("---")
    if st.button("🎯 Predict Missing Fields", type="primary"):
        try:
            input_data = {
                "Target": target,
                "Metal source (mmol)": metal_source,
                "Linker(s) (mmol)": linker,
                "Solvent(s) (mL)": solvent,
                "Modulator / Additive": modulator,
                "Pressure": pressure,
                "Method": method,
                "Wash / Activation": wash_activation
            }
            input_data = {k: v for k, v in input_data.items() if isinstance(v, str) and v.strip()}

            predictions = predict_missing_fields(input_data, models)

            complete_row = input_data.copy()
            complete_row.update(predictions)
            for field in COLUMN_NAMES:
                if field not in complete_row:
                    complete_row[field] = ""

            st.session_state.predicted_data.append(complete_row)
            st.success("✅ Predictions completed!")

            if predictions:
                st.subheader("📈 Predictions")
                cols = st.columns(len(predictions))
                for i, (field, value) in enumerate(predictions.items()):
                    with cols[i]:
                        st.metric(field, value)

            st.subheader("📊 Complete Synthesis Protocol")
            display_mof_row(complete_row)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    if st.session_state.predicted_data:
        st.markdown("---")
        st.subheader("📊 Predicted Data")
        st.markdown(f"*{len(st.session_state.predicted_data)} prediction(s) completed*")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = download_csv(st.session_state.predicted_data)
            st.download_button("📥 Download CSV", csv_data, "predicted_mof_data.csv", "text/csv", key="predict_csv_dl", use_container_width=True)
        with c2:
            json_data = download_json(st.session_state.predicted_data)
            st.download_button("📥 Download JSON", json_data, "predicted_mof_data.json", "application/json", key="predict_json_dl", use_container_width=True)
        df = pd.DataFrame(st.session_state.predicted_data)
        st.dataframe(df, use_container_width=True)

def suggest_tab() -> None:
    st.markdown(
        "<div class='mof-card'><h2>💡 Suggest Synthesis Protocol</h2><p>Generate complete MOF synthesis protocols from partial cues.</p></div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("### 💭 Input Synthesis Cues")

    cues_input = st.text_area(
        "Enter synthesis cues:",
        height=150,
        placeholder="e.g., Gold-based MOF, room temperature, antimicrobial properties, DMF solvent..."
    )

    st.markdown("---")
    if st.button("✨ Generate Protocol", type="primary") and cues_input.strip():
        # Validate input
        if not validate_text_input(cues_input, max_length=2000, field_name="Synthesis cues"):
            return
        
        # Check rate limit
        if not check_rate_limit("suggest", max_requests=10, window_seconds=60):
            return
        
        if not validate_environment():
            return

        try:
            from openai import OpenAI
            client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return

        try:
            with st.spinner("Generating synthesis protocol..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_suggestion_prompt(cues_input)}
                ]
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=RESPONSE_FORMAT,
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
                suggested_row = parse_json_response(response_text)

                if suggested_row and validate_mof_row(suggested_row):
                    cleaned_row = clean_mof_row(suggested_row)
                    st.session_state.suggested_data.append(cleaned_row)
                    st.success("✅ Successfully generated synthesis protocol!")
                    display_mof_row(cleaned_row)
                else:
                    st.error("❌ Failed to generate valid synthesis protocol")

        except Exception as e:
            st.error(f"❌ Suggestion failed: {e}")

    if st.session_state.suggested_data:
        st.markdown("---")
        st.subheader("📊 Suggested Protocols")
        st.markdown(f"*{len(st.session_state.suggested_data)} protocol(s) suggested*")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = download_csv(st.session_state.suggested_data)
            st.download_button("📥 Download CSV", csv_data, "suggested_mof_data.csv", "text/csv", key="suggest_csv_dl", use_container_width=True)
        with c2:
            json_data = download_json(st.session_state.suggested_data)
            st.download_button("📥 Download JSON", json_data, "suggested_mof_data.json", "application/json", key="suggest_json_dl", use_container_width=True)
        df = pd.DataFrame(st.session_state.suggested_data)
        st.dataframe(df, use_container_width=True)

# ===== Main =====
def main():
    st.markdown(
        """
        <div class="mof-card">
            <h1 style="margin-bottom: 0.4rem;">🧪 MOF Synthesis Assistant</h1>
            <p style="color:#cbd5f5; max-width: 720px;">
                An interactive assistant for designing, extracting, and predicting MOF synthesis protocols.
                Use the tabs below to go from free text or minimal cues to structured, exportable synthesis tables.
            </p>
        </div>
        <div class="mof-card" style="margin-top: -0.4rem;">
            <h3 style="margin-top: 0;">Modes</h3>
            <ul style="padding-left: 1.1rem;">
                <li>⚗️ <strong>Complete (Metal + Linker)</strong>: give minimal inputs → get a full protocol (GPT)</li>
                <li>🔍 <strong>Extract</strong>: extract structured data from free text (GPT)</li>
                <li>🔮 <strong>Predict</strong>: predict missing fields (shows only if ML models exist)</li>
                <li>💡 <strong>Suggest</strong>: generate a protocol from free-form cues (GPT)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    tabs = []
    # Always show the new Quick Complete tab
    tabs.append(("⚗️ Complete (Metal + Linker)", quick_complete_tab))
    tabs.append(("🔍 Extract", extract_tab))
    if models_available():
        tabs.append(("🔮 Predict", predict_tab))
    tabs.append(("💡 Suggest", suggest_tab))

    # Build tabs dynamically
    st_tabs = st.tabs([t[0] for t in tabs])
    for tab, fn in zip(st_tabs, [t[1] for t in tabs]):
        with tab:
            fn()

    with st.sidebar:
        st.markdown("<div style='text-align: center; padding: 1rem 0;'><h2>🔧 Configuration</h2></div>", unsafe_allow_html=True)
        
        st.markdown("#### Environment")
        if validate_environment():
            st.success("✅ OPENAI_API_KEY configured")
        else:
            st.error("❌ Missing OPENAI_API_KEY")
        
        st.markdown("---")
        st.markdown("#### Session Data")
        total_items = (
            len(st.session_state.get('extracted_data', [])) +
            len(st.session_state.get('predicted_data', [])) +
            len(st.session_state.get('suggested_data', [])) +
            len(st.session_state.get('completed_data', []))
        )
        st.info(f"📊 {total_items} total item(s) in session")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.extracted_data = []
            st.session_state.predicted_data = []
            st.session_state.suggested_data = []
            st.session_state.completed_data = []
            st.success("✨ Data cleared!")
        
        st.markdown("---")
        st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #94a3b8; margin-top: 2rem;'>MOF Synthesis Assistant v1.0<br/>Powered by GPT-4 & XGBoost</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
