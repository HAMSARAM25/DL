import os
import sys

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# ======================================
# FIX PYTHON PATHS
# ======================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "../.."))

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

from explainability.captum_explainer import captum_explain_single
# ======================================
# IMPORT LOCAL MODULES
# ======================================
from explainability.gradcam import generate_gradcam
from explainability.slice_visualizer import visualize_slices
from models.hybrid_model import HybridMRIModel
from treatment.recommend import recommend_treatment

# ======================================
# STREAMLIT CONFIG
# ======================================
st.set_page_config(
    page_title="MRI Dementia Classifier",
    page_icon="🧠",
    layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

# ======================================
# LOAD MODEL (CACHED)
# ======================================
@st.cache_resource
def load_model():
    model = HybridMRIModel(num_classes=4)
    model_path = os.path.join(ROOT_DIR, "saved_models/hybrid_mri_model.pth")

    if not os.path.exists(model_path):
        st.error("❌ Model not found at saved_models/hybrid_mri_model.pth")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

model = load_model()

# ======================================
# IMAGE TRANSFORMS
# ======================================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======================================
# SIDEBAR
# ======================================
st.sidebar.title("⚙️ Model Settings")
st.sidebar.markdown("---")
opacity = st.sidebar.slider("Grad-CAM Opacity", 0.1, 1.0, 0.6, 0.1)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
show_ig = st.sidebar.checkbox("Show IG Explanation", value=True)
st.sidebar.markdown("---")
st.sidebar.info("Hybrid CNN-RNN-ViT Dementia Classifier")

# ======================================
# MAIN UI
# ======================================
st.title("🧠 Alzheimer's Disease MRI  Classification & Treatment Advisor")
st.write("Upload MRI → AI predicts dementia stage → Provides explainability → Shows treatment guidance.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# ======================================
# WHEN USER UPLOADS IMAGE
# ======================================
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = tf(img).unsqueeze(0).to(device)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼 Uploaded MRI")
        st.image(img, width=350)

    # ======= Prediction =======
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    with col2:
        st.subheader("📊 Model Predicted Result")
        st.markdown(f"""
        <div style="padding:20px; background:#202020; border-radius:10px;">
            <h3 style="color:#4fe34f;">Prediction: {pred_class}</h3>
            <p style="font-size:18px;">Confidence: <b>{confidence:.4f}</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================
    # TABS (Grad-CAM / IG / Treatment)
    # ======================================
    tab1, tab2, tab3 = st.tabs(["🔥 Grad-CAM", "🧠 Integrated Gradients", "🏥 Treatment Plan"])

    # ---------------------------------------------------
    # TAB 1 — GRAD-CAM
    # ---------------------------------------------------
    with tab1:
        if show_gradcam:
            st.subheader("🔥 Grad-CAM Heatmap")
            try:
                target_layer = model.cnn.layer4[-1]
                cam = generate_gradcam(model, img_tensor, img, target_layer)

                overlay = (
                    cam * opacity +
                    np.array(img.resize((224, 224))) * (1-opacity)
                ).astype(np.uint8)

                st.image(overlay, width=350, caption="Grad-CAM Overlay")

            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")

    # ---------------------------------------------------
    # TAB 2 — INTEGRATED GRADIENTS (CAPTUM)
    # ---------------------------------------------------
    with tab2:
        if show_ig:
            st.subheader("🧠 Integrated Gradients (Captum)")

            try:
                with st.spinner("Computing Integrated Gradients..."):
                    ig_map = captum_explain_single(model, img_tensor, pred_idx)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(ig_map, cmap="inferno")
                fig.colorbar(ax.images[0], ax=ax)
                ax.set_title("Integrated Gradients Heatmap")
                ax.axis("off")

                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"IG Error: {e}")

    # ---------------------------------------------------
    # TAB 3 — TREATMENT PLAN
    # ---------------------------------------------------
    with tab3:
        st.subheader("🏥 Personalized Treatment Recommendation")

        treatment = recommend_treatment(pred_class)

        # ===== Diagnosis Card =====
        st.markdown(f"""
            <div style="
                padding:20px;
                background-color:#2c2c2c;
                border-radius:12px;
                border-left:6px solid #4ee44e;
                margin-bottom:20px;">
                <h3 style="margin-top:0; color:#4ee44e;">🧠 Diagnosis</h3>
                <p style="font-size:18px; margin:0;"><b>{pred_class}</b></p>
            </div>
        """, unsafe_allow_html=True)

        # ===== Treatment Card =====
        if "Treatment" in treatment:
            st.markdown("""
                <div style="
                    padding:20px;
                    background-color:#1f1f1f;
                    border-radius:12px;
                    border-left:6px solid #ff9f43;
                    margin-bottom:20px;">
                    <h3 style="margin-top:0; color:#ff9f43;">💊 Treatment Options</h3>
            """, unsafe_allow_html=True)

            for item in treatment["Treatment"]:
                st.markdown(f"- ✔ <span style='font-size:17px;'>{item}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ===== Recommendations Card =====
        if "Recommendations" in treatment:
            st.markdown("""
                <div style="
                    padding:20px;
                    background-color:#1f1f1f;
                    border-radius:12px;
                    border-left:6px solid #3498db;
                    margin-bottom:20px;">
                    <h3 style="margin-top:0; color:#3498db;">📋 Recommendations</h3>
            """, unsafe_allow_html=True)

            for item in treatment["Recommendations"]:
                st.markdown(f"- ✔ <span style='font-size:17px;'>{item}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
