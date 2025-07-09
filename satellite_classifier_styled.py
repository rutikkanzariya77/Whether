import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# --- Constants (define first to avoid NameError) ---
COLOR_THEME = {
    'primary': '#4169E1',      # Royal Blue
    'secondary': '#32CD32',    # Lime Green
    'background': '#1e1e2f',
    'text': '#f8f9fa',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545'
}

# --- Page Config ---
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🌍",
    layout="wide"
)

# --- Custom CSS ---
st.markdown(f"""
<style>
    body {{ background-color: {COLOR_THEME['background']}; color: {COLOR_THEME['text']}; }}
    .main-header {{
        background: linear-gradient(135deg, {COLOR_THEME['primary']} 0%, {COLOR_THEME['secondary']} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }}
    .card {{
        background: #2c2f3a;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        color: {COLOR_THEME['text']};
    }}
    .color-display {{
        width: 100%;
        height: 40px;
        border-radius: 6px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }}
    [data-testid="stFileUploader"] > div {{
        background-color: #1e1e2f !important;
        border: 2px dashed #4a4a6a !important;
        color: #f8f9fa !important;
        border-radius: 12px;
        padding: 1.5rem;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: {COLOR_THEME['danger']} !important;
        color: white !important;
        border: none;
        border-radius: 6px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Dummy Prediction Function ---
def mock_predict(image_array, class_names):
    np.random.seed(42)
    predictions = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    idx = np.argmax(predictions)
    return class_names[idx], predictions[idx], predictions

# --- Image Preprocessing ---
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Class Labels (use mock or load from CSV) ---
@st.cache_data
def load_class_labels():
    default_classes = [
        'Agricultural Land', 'Airplane', 'Beach', 'Buildings', 'Forest',
        'Freeway', 'Harbor', 'River', 'Runway', 'Tennis Court'
    ]
    return default_classes

# --- Main UI ---
def main():
    st.markdown('<div class="main-header"><h1>🌍 Satellite Image Classifier</h1><p>Classify land types using AI</p></div>', unsafe_allow_html=True)

    class_names = load_class_labels()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Image")
        uploaded_file = st.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Classify Image"):
                st.toast("Analyzing image...", icon="🧠")
                processed = preprocess_image(image)
                pred_class, conf, all_preds = mock_predict(processed, class_names)
                st.session_state['results'] = {
                    'class': pred_class,
                    'confidence': conf,
                    'all_preds': all_preds,
                    'classes': class_names
                }
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Prediction Results")

        if 'results' in st.session_state:
            result = st.session_state['results']
            st.success(f"Predicted: {result['class']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.markdown("### Top Predictions")
            prob_df = pd.DataFrame({
                'Class': result['classes'],
                'Probability': result['all_preds']
            }).sort_values('Probability', ascending=False).head(5)

            st.bar_chart(prob_df.set_index('Class'))
        else:
            st.info("Upload an image and click 'Classify Image' to see results.")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
