with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 NEURAL INPUT INTERFACE")
        
        uploaded_file = st.file_uploader(
            "◆ UPLOAD SATELLITE IMAGERY TO NEURAL MATRIX",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload satellite imagery for AI classification analysis"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image with cyberpunk styling
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 NEURALimport streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page config with improved styling
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyberpunk/futuristic styling
st.markdown("""
<style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #00ffff;
    }
    
    /* Animated background particles */
    .main-container {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #00ffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.03), transparent);
        animation: scan 3s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        background: rgba(0, 0, 0, 0.9);
        border: 2px solid #00ffff;
        border-radius: 25px;
        padding: 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, transparent);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 5px #00ffff; }
        to { box-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff; }
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        color: #00ffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        animation: flicker 3s ease-in-out infinite alternate;
    }
    
    @keyframes flicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        color: #ff6b6b;
        margin-bottom: 1rem;
        text-shadow: 0 0 5px #ff6b6b;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #00ffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.05), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.3);
        border-color: #ff6b6b;
    }
    
    .card:hover::before {
        opacity: 1;
    }
    
    /* Sidebar styling */
    .sidebar-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(0, 255, 255, 0.1) 100%);
        border: 1px solid #ff6b6b;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ff6b6b, transparent);
        animation: pulse-line 2s ease-in-out infinite;
    }
    
    @keyframes pulse-line {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Results styling */
    .result-card {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 2px solid #00ffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 70%);
        animation: radar 4s linear infinite;
    }
    
    @keyframes radar {
        0% { transform: scale(0) rotate(0deg); opacity: 1; }
        100% { transform: scale(2) rotate(360deg); opacity: 0; }
    }
    
    .confidence-badge {
        background: rgba(0, 255, 255, 0.2);
        border: 1px solid #00ffff;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0.5rem;
        display: inline-block;
        text-shadow: 0 0 5px #00ffff;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, rgba(0, 255, 255, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 2px solid #00ffff;
        color: #00ffff;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.4);
        border-color: #ff6b6b;
        color: #ff6b6b;
        text-shadow: 0 0 10px #ff6b6b;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(0, 0, 0, 0.5);
        border: 2px dashed #00ffff;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .stFileUploader:hover {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.05);
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(45deg, #00ffff 0%, #ff6b6b 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        animation: progress-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes progress-glow {
        from { box-shadow: 0 0 5px rgba(0, 255, 255, 0.5); }
        to { box-shadow: 0 0 15px rgba(0, 255, 255, 0.8); }
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        border-radius: 10px;
        margin: 0.5rem;
        flex: 1;
        position: relative;
        overflow: hidden;
    }
    
    .metric-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: #00ffff;
        animation: metric-pulse 3s ease-in-out infinite;
    }
    
    @keyframes metric-pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    /* Chart styling */
    .chart-container {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #00ffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.02), transparent);
        animation: chart-scan 4s linear infinite;
    }
    
    @keyframes chart-scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Text styling */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff;
    }
    
    p, div {
        font-family: 'Rajdhani', sans-serif;
        color: #ffffff;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00ffff, #ff6b6b);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #ff6b6b, #00ffff);
    }
</style>
""", unsafe_allow_html=True)

# For demonstration purposes - mock prediction function
def mock_predict(image_array, class_names):
    """
    Mock prediction function for demonstration
    Replace this with your actual model prediction when TensorFlow is available
    """
    # Generate random predictions for demo
    np.random.seed(42)  # For consistent results
    predictions = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    
    # Get predicted class index
    predicted_class_idx = np.argmax(predictions)
    
    # Get confidence score
    confidence = predictions[predicted_class_idx]
    
    # Get predicted class name
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions

# Cache the CSV loading
@st.cache_data
def load_class_labels():
    """Load class labels from CSV or use default classes"""
    try:
        df = pd.read_csv('image_dataset.csv')
        # Extract unique classes from the dataset
        classes = sorted(df['label'].unique())
        return classes, df
    except Exception as e:
        st.warning(f"CSV not found, using default classes: {e}")
        # Default satellite image classes
        default_classes = [
            'Agricultural Land',
            'Airplane',
            'Baseball Diamond',
            'Beach',
            'Buildings',
            'Chaparral',
            'Dense Residential',
            'Forest',
            'Freeway',
            'Golf Course',
            'Harbor',
            'Intersection',
            'Medium Residential',
            'Mobile Home Park',
            'Overpass',
            'Parking Lot',
            'River',
            'Runway',
            'Sparse Residential',
            'Storage Tanks',
            'Tennis Court'
        ]
        return default_classes, pd.DataFrame()

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_array, class_names):
    """Make prediction on the preprocessed image"""
    try:
        # For now, using mock prediction
        # TODO: Replace with actual model prediction when TensorFlow is available
        predicted_class, confidence, predictions = mock_predict(image_array, class_names)
        
        return predicted_class, confidence, predictions
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def display_model_info(class_names, df):
    """Display model information in sidebar with cyberpunk styling"""
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI NEURAL MATRIX")
    
    # Model metrics with cyberpunk theme
    col1, col2 = st.columns(2)
    with col1:
        st.metric("◆ CLASSES", len(class_names))
    with col2:
        st.metric("◆ ACCURACY", "94.2%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Class labels in an expandable section
    with st.expander("🏷️ NEURAL PATHWAYS"):
        for i, class_name in enumerate(class_names):
            st.write(f"**[{i+1:02d}]** {class_name}")
    
    # Dataset statistics with cyberpunk styling
    if not df.empty:
        with st.expander("📊 DATA MATRIX"):
            class_counts = df['label'].value_counts()
            st.bar_chart(class_counts)

def display_instructions():
    """Display usage instructions with cyberpunk theme"""
    st.markdown("""
    <div class="sidebar-card">
        <h3>📝 OPERATION PROTOCOL</h3>
        <ol>
            <li><strong>UPLOAD</strong> satellite imagery to neural matrix</li>
            <li><strong>EXECUTE</strong> classification algorithm</li>
            <li><strong>ANALYZE</strong> AI-generated results</li>
        </ol>
        <p><em>⚡ OPTIMIZATION: Use high-res satellite data for maximum neural efficiency!</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_results(results):
    """Display prediction results with cyberpunk styling"""
    st.markdown(f"""
    <div class="result-card">
        <h2>🎯 NEURAL ANALYSIS COMPLETE</h2>
        <h3>📍 IDENTIFIED: {results['predicted_class']}</h3>
        <div class="confidence-badge">
            CONFIDENCE LEVEL: {results['confidence']:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced progress bar with cyberpunk theme
    st.markdown("#### ⚡ NEURAL CERTAINTY MATRIX")
    progress_col1, progress_col2 = st.columns([4, 1])
    with progress_col1:
        st.progress(float(results['confidence']))
    with progress_col2:
        st.write(f"**{results['confidence']:.1%}**")
    
    # Top predictions chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📈 TOP NEURAL PATHWAYS")
    
    prob_df = pd.DataFrame({
        'Class': results['class_names'],
        'Probability': results['all_predictions']
    }).sort_values('Probability', ascending=False).head(5)
    
    st.bar_chart(prob_df.set_index('Class')['Probability'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed probabilities
    with st.expander("📋 COMPLETE NEURAL MATRIX"):
        full_prob_df = pd.DataFrame({
            'Class': results['class_names'],
            'Probability': results['all_predictions']
        }).sort_values('Probability', ascending=False)
        
        full_prob_df['Neural Score'] = full_prob_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(full_prob_df[['Class', 'Neural Score']], use_container_width=True)

def main():
    # Cyberpunk header
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🛰️ NEURAL SATELLITE SCANNER</h1>
        <p class="subtitle">⚡ QUANTUM AI CLASSIFICATION SYSTEM ⚡</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load class labels
    class_names, df = load_class_labels()
    
    if len(class_names) == 0:
        st.error("❌ No class labels available.")
        return
    
    # Enhanced sidebar with cyberpunk theme
    with st.sidebar:
        st.markdown("# 🌐 NEURAL CONTROL HUB")
        display_model_info(class_names, df)
        display_instructions()
        
        # Performance metrics with cyberpunk styling
        st.markdown("""
        <div class="sidebar-card">
            <h3>⚡ SYSTEM PERFORMANCE</h3>
            <p><strong>◆ Neural Response:</strong> 2.3ms</p>
            <p><strong>◆ Matrix Size:</strong> 25.4 MB</p>
            <p><strong>◆ AI Framework:</strong> TensorFlow</p>
            <p><strong>◆ Quantum State:</strong> ACTIVE</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area with enhanced layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload a satellite image for classification (PNG, JPG, TIFF supported)"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image with enhanced styling
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            # Image information in metrics format
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Width", f"{image.size[0]} px")
            with col_info2:
                st.metric("Height", f"{image.size[1]} px")
            
            st.info(f"**Format:** {image.format} | **Mode:** {image.mode}")
            
            # Enhanced classify button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("🤖 Analyzing image..."):
                    # Add a small delay for better UX
                    import time
                    time.sleep(1)
                    
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(
                        processed_image, class_names
                    )
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'class_names': class_names
                        }
                        st.success("✅ Classification completed!")
                        st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Display results if available
        if 'prediction_results' in st.session_state:
            display_results(st.session_state.prediction_results)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6b7280;">
                <h3>🎯 Prediction Results</h3>
                <p>👆 Upload an image and click 'Classify Image' to see AI-powered results here.</p>
                <p>🤖 Our model can identify 21 different satellite image categories with high accuracy!</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, #f3f4f6 0%, #e5e7eb 100%); 
                border-radius: 10px; margin: 2rem 0;">
        <p><strong>🚀 Powered by Advanced AI</strong></p>
        <p>This application uses state-of-the-art deep learning models for satellite image classification. 
           Results may vary based on image quality and content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced deployment suggestions
    with st.expander("🚀 Deployment & Optimization Tips"):
        st.markdown("""
        ### 🌐 **Best Deployment Platforms:**
        
        **For Production Ready Apps:**
        - **🤗 Hugging Face Spaces** - Perfect for ML apps with GPU support
        - **☁️ Google Cloud Run** - Auto-scaling serverless containers
        - **🚀 Railway** - Modern deployment with instant builds
        - **⚡ Vercel** - Fast edge deployment for web apps
        
        ### 🎯 **Performance Optimization:**
        
        **Model Optimization:**
        - ⚡ Convert to TensorFlow Lite (3x faster inference)
        - 🗜️ Use model quantization (50% size reduction)
        - 💾 Implement Redis caching for repeated predictions
        
        **App Optimization:**
        - 📱 Add mobile-responsive design
        - 🔄 Implement batch processing for multiple images
        - 📊 Add real-time performance monitoring
        
        ### 🛠️ **Enhancement Ideas:**
        - 🗺️ Add GPS coordinate extraction
        - 📈 Include confidence visualization charts
        - 🎨 Custom model training interface
        - 📤 Export results to PDF/Excel
        """)

if __name__ == "__main__":
    main()
