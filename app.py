import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import tensorflow as tf
from PIL import Image
import numpy as np
from google import genai
from huggingface_hub import hf_hub_download

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Multi-Modal Healthcare AI", page_icon="🩺", layout="wide")

# --- HIDE STREAMLIT DEPLOY BUTTON & MENU ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("🩺 Multi-Modal Medical Diagnostic Platform")

SKIN_CLASSES = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease',
    'Cellulitis', 'Eczema', 'Exanthems', 'Hair Loss', 'Herpes STDs',
    'Pigmentation Disorders', 'Lupus', 'Melanoma Moles', 'Nail Fungus',
    'Poison Ivy', 'Psoriasis', 'Scabies Lyme', 'Seborrheic Keratoses',
    'Systemic Disease', 'Ringworm Candidiasis', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis', 'Warts Viral Infections'
]
EMG_CLASSES = ['Healthy', 'Neuropathy', 'Myopathy']

HF_REPO_ID = "BabaYaga1514/healthcare-models"




# Cache Model Loaders
@st.cache_resource
def load_skin_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(SKIN_CLASSES)))
    path = "skin_resnet50_best.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


@st.cache_resource
def load_emg_model():
    path = "emg_cnn_lstm_best.keras"
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None


skin_model = load_skin_model()
emg_model = load_emg_model()

tab1, tab2, tab3 = st.tabs(["🔍 Skin Disease Detection", "📈 EMG Signal Analysis", "💬 Symptom AI Assistant"])

# --- TAB 1: SKIN DETECTION ---
with tab1:
    st.header("Visual Dermatological Screening (ResNet-50)")
    uploaded_file = st.file_uploader("Upload skin lesion image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300)
        if st.button("Analyze Image"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = skin_model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_idx = torch.topk(probs, 1)

            st.success(f"**Predicted Condition:** {SKIN_CLASSES[top_idx[0]]} ({top_prob[0].item() * 100:.2f}%)")

# --- TAB 2: EMG SIGNAL ANALYSIS ---
with tab2:
    st.header("EMG Signal Classification (CNN-LSTM)")
    emg_file = st.file_uploader("Upload 1D raw EMG CSV signal", type=["csv", "npy"])
    if emg_file:
        try:
            signal = np.loadtxt(emg_file, delimiter=',') if emg_file.name.endswith('.csv') else np.load(emg_file)
            st.line_chart(signal[:300])
            if st.button("Run Signal Analysis"):
                if emg_model:
                    sample = signal[:1000].reshape(1, 1000, 1)
                    preds = emg_model.predict(sample)
                    cls_idx = np.argmax(preds)
                    st.info(f"**Neuromuscular Result:** {EMG_CLASSES[cls_idx]} ({np.max(preds) * 100:.2f}%)")
                else:
                    st.error("Model weights file 'emg_cnn_lstm_best.keras' missing from local directory.")
        except Exception as e:
            st.error(f"Error reading signal array: {e}")

# --- TAB 3: SYMPTOM ASSISTANT ---
with tab3:
    st.header("Symptom Assistant")

    # Read securely from environment or fallback to input box
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Enter Gemini API Key:", type="password")

    user_input = st.text_area("Describe symptoms in plain text:")
    if st.button("Analyze Symptoms"):
        if api_key and user_input:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=f"Context: Triage Assistant. Symptoms: {user_input}. Provide preliminary guidance."
            )
            st.write(resp.text)
        elif not api_key:
            st.error("Please provide a valid Gemini API key.")