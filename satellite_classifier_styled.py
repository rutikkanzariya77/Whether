import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import time

# Constants for the UI
COLOR_THEME = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#2ca02c',    # Green
    'background': '#0d1117',   # Dark background
    'text': '#c9d1d9',         # Light text
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545'
}

# Set Streamlit page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_THEME['background']};
        color: {COLOR_THEME['text']};
    }}

    .main-header {{
        background: linear-gradient(135deg, {COLOR_THEME['primary']} 0%, {COLOR_THEME['secondary']} 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }}

    .card {{
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }}

    .color-display {{
        width: 100%;
        height: 40px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-weight: bold;
    }}

    [data-testid="stFileUploader"] > div {{
        background-color: #21262d;
        border: 2px dashed #3b3f46;
        border-radius: 12px;
        padding: 1.5rem;
        color: #f0f6fc;
    }}

    [data-testid="stFileUploader"] button {{
        background-color: {COLOR_THEME['danger']};
        color: white;
        border-radius: 6px;
    }}
</style>
""", unsafe_allow_html=True)

# Mock prediction function
def mock_predict(image_array, class_names):
    np.random.seed(42)
    predictions = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    predicted_class_idx = np.argmax(predictions)
    return class_names[predicted_class_idx], predictions[predicted_class_idx], predictions

# Load class labels
def load_class_labels():
    try:
        df = pd.read_csv("image_dataset.csv")
        classes = sorted(df['label'].unique())
        return classes, df
    except:
        default_classes = [
            'Forest', 'Water', 'Desert', 'Urban', 'Agricultural',
            'Snow', 'Mountain', 'Beach', 'River', 'Cloud'
        ]
        return default_classes, pd.DataFrame()

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict image
def predict_image(image_array, class_names):
    return mock_predict(image_array, class_names)

# Display result section
def display_results(results):
    st.markdown("""
    <div class="card">
        <h2>🌟 Prediction Results</h2>
        <h3>🔹 Class: <span style="color:#58a6ff">{}</span></h3>
        <p>Confidence: <strong>{:.1%}</strong></p>
    </div>
    """.format(results['predicted_class'], results['confidence']), unsafe_allow_html=True)

    st.progress(float(results['confidence']))

    df = pd.DataFrame({
        'Class': results['class_names'],
        'Probability': results['all_predictions']
    }).sort_values('Probability', ascending=False)

    st.bar_chart(df.set_index('Class'))

# Main UI
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ Satellite Image Classifier</h1>
        <p>AI-powered Land Cover Detection</p>
    </div>
    """, unsafe_allow_html=True)

    class_names, df = load_class_labels()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload Your Image")

        uploaded_file = st.file_uploader("Choose a satellite image...", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Classify Image"):
                with st.spinner("Analyzing image..."):
                    time.sleep(1)
                    img_array = preprocess_image(image)
                    predicted_class, confidence, all_predictions = predict_image(img_array, class_names)

                    st.session_state.prediction_results = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'class_names': class_names
                    }

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if 'prediction_results' in st.session_state:
            display_results(st.session_state.prediction_results)
        else:
            st.info("Upload an image and click 'Classify Image' to see results.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:1rem; color:#8b949e">
        🚀 Powered by Deep Learning | Streamlit App
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
