# ============================================================
# app/app.py
# PURPOSE: Professional Streamlit web application for
#          skin lesion classification
# RUN: streamlit run app/app.py (from project root)
# ============================================================

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# Add project root to path so we can import predict.py and config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from predict import (
    load_model_cached,
    preprocess_image,
    preprocess_metadata,
    predict,
    get_risk_level,
    get_disease_description
)
import config

# ============================================================
# SECTION 1 — PAGE CONFIGURATION
# Must be the FIRST streamlit command in the file
# ============================================================
st.set_page_config(
    page_title="DermAI — Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SECTION 2 — CUSTOM CSS STYLING
# Makes the app look professional and modern
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }

    /* Prediction result card */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
    }

    /* Risk cards */
    .risk-high {
        background: #fef2f2;
        border: 2px solid #fecaca;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .risk-moderate {
        background: #fff7ed;
        border: 2px solid #fed7aa;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .risk-low {
        background: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Disclaimer box */
    .disclaimer {
        background: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #78350f;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 3 — LOAD MODEL (cached — runs only once)
# ============================================================
model = None
encoder = None
model_loaded = False
model_error = None

try:
    model, encoder = load_model_cached()
    model_loaded = True
except FileNotFoundError as e:
    model_error = str(e)
except Exception as e:
    model_error = f"Unexpected error loading model: {e}"

# ============================================================
# SECTION 4 — HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🔬 DermAI: Skin Lesion Classifier</h1>
    <p style="margin:0.5rem 0 0; color:#cbd5e1; font-size:1rem;">
        AI-Powered Multi-Modal Skin Cancer Detection using
        EfficientNetB0 + Clinical Metadata Fusion
    </p>
</div>
""", unsafe_allow_html=True)

# Show model status
if model_loaded:
    st.success("✅ Model loaded and ready for predictions")
else:
    st.error(f"❌ Model not loaded: {model_error}")
    st.info(
        "**How to fix:** Complete training on Google Colab, "
        "then copy `saved_models/best_model.h5` and "
        "`saved_models/localization_encoder.pkl` to your local project."
    )

# ============================================================
# SECTION 5 — SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🔬 DermAI")
    st.markdown("---")

    st.markdown("### 📋 About This Tool")
    st.markdown("""
    - AI model trained on **HAM10000** dataset
    - **10,015** dermoscopic images
    - Detects **7 skin lesion types**
    - Uses **image + patient metadata**
    - Model: **EfficientNetB0** + Metadata Fusion
    """)

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Upload a dermoscopic image
    2. Enter patient age, gender, location
    3. Click **Analyze Lesion**
    4. Review prediction and risk level
    """)

    st.markdown("---")
    st.markdown("### 🏷️ Detectable Classes")
    classes_info = {
        "🔴 Melanoma": "mel",
        "🟤 Melanocytic Nevi": "nv",
        "🔴 Basal Cell Carcinoma": "bcc",
        "🟠 Actinic Keratoses": "akiec",
        "🟢 Benign Keratosis": "bkl",
        "🟢 Dermatofibroma": "df",
        "🟢 Vascular Lesions": "vasc"
    }
    for name in classes_info:
        st.markdown(f"- {name}")

    st.markdown("---")
    st.markdown("""
    <div style="background:#1e293b; padding:1rem; border-radius:8px;
                border-left:3px solid #ef4444;">
        <strong>⚠️ DISCLAIMER</strong><br>
        This tool is for <strong>educational purposes only</strong>.
        It does NOT replace professional medical diagnosis.
        Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 6 — MAIN LAYOUT (Two Columns)
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")

# ============================================================
# LEFT COLUMN — INPUT
# ============================================================
with left_col:
    st.markdown("### 📷 Upload Skin Lesion Image")

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermoscopic image of the skin lesion"
    )

    if uploaded_file:
        # Display uploaded image
        img_display = Image.open(uploaded_file)
        st.image(
            img_display,
            caption="Uploaded Lesion Image",
            use_column_width=True
        )
        # Show image info
        st.caption(
            f"Size: {img_display.size[0]}x{img_display.size[1]}px | "
            f"Format: {uploaded_file.type}"
        )

    st.markdown("---")
    st.markdown("### 👤 Patient Information")

    # Age input
    age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=100,
        value=45,
        help="Enter patient age in years"
    )

    # Gender input
    sex = st.selectbox(
        "Gender",
        options=['Male', 'Female', 'Unknown'],
        help="Select patient gender"
    )

    # Localization input
    localization = st.selectbox(
        "Lesion Location",
        options=config.LOCALIZATION_CATEGORIES,
        help="Select the body location of the lesion"
    )

    st.markdown("---")

    # Predict button
    analyze_clicked = st.button(
        "🔍 Analyze Lesion",
        use_container_width=True
    )

# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================
with right_col:
    st.markdown("### 📊 Analysis Results")

    if not analyze_clicked:
        # Show placeholder when no prediction yet
        st.markdown("""
        <div style="background:#f1f5f9; border-radius:12px; padding:3rem;
                    text-align:center; color:#94a3b8;">
            <div style="font-size:4rem;">🔬</div>
            <h3 style="color:#94a3b8;">Awaiting Analysis</h3>
            <p>Upload an image and fill in patient details,
            then click <strong>Analyze Lesion</strong></p>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded_file:
        # User clicked but forgot to upload image
        st.warning("⚠️ Please upload a skin lesion image first.")

    elif not model_loaded:
        # Model not available
        st.error("❌ Model is not loaded. Please check the error above.")

    else:
        # ====================================================
        # RUN PREDICTION
        # ====================================================
        with st.spinner("🔄 Analyzing lesion... Please wait..."):
            try:
                # Reset file pointer before reading
                uploaded_file.seek(0)

                # Preprocess image
                image_array = preprocess_image(uploaded_file)

                # Preprocess metadata
                metadata_array = preprocess_metadata(
                    age, sex, localization, encoder
                )

                # Run prediction
                results = predict(model, image_array, metadata_array)

                # Get risk level
                risk = get_risk_level(results['predicted_class'])

                # Get disease description
                disease_info = get_disease_description(
                    results['predicted_class']
                )

                prediction_success = True

            except Exception as e:
                prediction_success = False
                prediction_error = str(e)

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================
        if prediction_success:

            # --- RESULT CARD 1: PREDICTION ---
            st.markdown(f"""
            <div class="result-card">
                <p style="color:#64748b; margin:0; font-size:0.85rem;
                          font-weight:600; text-transform:uppercase;">
                    Predicted Diagnosis
                </p>
                <h2 style="color:#0f172a; margin:0.3rem 0;">
                    {results['predicted_label']}
                </h2>
                <p style="color:#475569; margin:0;">
                    Confidence:
                    <strong style="color:#2563eb; font-size:1.1rem;">
                        {results['confidence']:.1f}%
                    </strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence progress bar
            st.progress(results['confidence'] / 100)

            # --- RESULT CARD 2: RISK LEVEL ---
            risk_class = (
                "risk-high" if "HIGH" in risk['level']
                else "risk-moderate" if "MODERATE" in risk['level']
                else "risk-low"
            )
            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="color:{risk['color']}; margin:0;">
                    {risk['emoji']} {risk['level']}
                </h3>
                <p style="margin:0.3rem 0 0; color:#374151;">
                    {risk['message']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # --- RESULT CARD 3: PROBABILITY CHART ---
            st.markdown("#### 📈 Confidence Across All Classes")

            # Get class names and probabilities
            class_names = list(results['all_probabilities'].keys())
            probs = list(results['all_probabilities'].values())

            # Highlight predicted class in blue, others in grey
            colors = [
                '#2563eb' if name == results['predicted_label']
                else '#e2e8f0'
                for name in class_names
            ]

            # Create horizontal bar chart using Plotly
            fig = go.Figure(go.Bar(
                x=probs,
                y=class_names,
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition='outside',
                hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                xaxis_title="Confidence (%)",
                xaxis=dict(range=[0, 115]),
                height=320,
                margin=dict(l=10, r=60, t=10, b=30),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- RESULT CARD 4: DISEASE INFORMATION ---
            with st.expander(
                f"ℹ️ About {results['predicted_label']}", expanded=False
            ):
                st.markdown(f"**Description:**")
                st.write(disease_info['description'])

                st.markdown(f"**Visual Symptoms:**")
                st.write(disease_info['symptoms'])

                st.markdown(f"**Recommended Action:**")
                st.write(disease_info['action'])

            # --- RESULT CARD 5: DISCLAIMER ---
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Medical Disclaimer:</strong>
                This prediction is generated by an AI model for
                <strong>educational purposes only</strong>.
                It does <strong>NOT</strong> constitute medical advice
                or replace professional diagnosis.
                Please consult a <strong>certified dermatologist</strong>
                for any medical concerns.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Show error if prediction failed
            st.error(f"❌ Prediction failed: {prediction_error}")
            st.info(
                "Common causes:\n"
                "- Model file is corrupted\n"
                "- Image format not supported\n"
                "- Try a different image"
            )