import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page config with custom styling
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .prediction-card h2 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .class-list {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
        margin: 1rem 0;
    }
    
    .class-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .class-item:last-child {
        border-bottom: none;
    }
    
    /* Upload area styling */
    .upload-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Results section styling */
    .results-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 2rem;
    }
    
    /* Hide Streamlit default styling */
    .stDeployButton {
        display: none;
    }
    
    .stDecoration {
        display: none;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
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

def main():
    # Main header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ Satellite Image Classifier</h1>
        <p>Advanced AI-powered satellite image classification using deep learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load class labels
    class_names, df = load_class_labels()
    
    if len(class_names) == 0:
        st.error("❌ No class labels available.")
        return
    
    # Enhanced sidebar with model info
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>📊 Model Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classes", len(class_names), delta=None)
        with col2:
            st.metric("Model", "CNN", delta="Active")
        
        # Display class names in a styled container
        st.markdown("### 🏷️ Classification Categories")
        class_list_html = '<div class="class-list">'
        for i, class_name in enumerate(class_names):
            class_list_html += f'<div class="class-item">🏷️ {class_name}</div>'
        class_list_html += '</div>'
        st.markdown(class_list_html, unsafe_allow_html=True)
        
        # Dataset statistics
        if not df.empty:
            st.markdown("### 📈 Dataset Statistics")
            class_counts = df['label'].value_counts()
            st.bar_chart(class_counts)
        
        # Instructions
        st.markdown("### 📝 How to Use")
        st.markdown("""
        <div class="info-card">
            <strong>Step 1:</strong> Upload a satellite image<br>
            <strong>Step 2:</strong> Click 'Classify Image'<br>
            <strong>Step 3:</strong> View detailed results<br><br>
            <em>💡 Tip: Higher resolution images provide better accuracy</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area with enhanced styling
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Your Image</h3>
            <p>Select a satellite image for AI classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload a satellite image for classification"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image with enhanced styling
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            # Image info in styled cards
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"""
                <div class="metric-container">
                    <strong>📏 Dimensions</strong><br>
                    {image.size[0]} × {image.size[1]} pixels
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="metric-container">
                    <strong>🎨 Color Mode</strong><br>
                    {image.mode}
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced predict button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your image..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(
                        processed_image, class_names
                    )
                    
                    if predicted_class is not None:
                        # Store results in session state for display in col2
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'class_names': class_names
                        }
                        st.success("✅ Classification complete!")
    
    with col2:
        st.markdown("""
        <div class="results-section">
            <h3>🎯 Classification Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display results if available
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Main prediction in styled card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 {results['predicted_class']}</h2>
                <div class="confidence-badge">
                    Confidence: {results['confidence']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(float(results['confidence']))
            
            # Show top predictions with enhanced styling
            st.markdown("### 📊 Detailed Analysis")
            
            # Create a DataFrame for better visualization
            prob_df = pd.DataFrame({
                'Class': results['class_names'],
                'Probability': results['all_predictions']
            }).sort_values('Probability', ascending=False)
            
            # Display top 5 predictions as metrics
            st.markdown("**🏆 Top 5 Predictions:**")
            for i, (idx, row) in enumerate(prob_df.head(5).iterrows()):
                if i < 3:  # Show top 3 with different styling
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div class="metric-container">
                        {medal} <strong>{row['Class']}</strong><br>
                        Probability: {row['Probability']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write(f"**{row['Class']}:** {row['Probability']:.1%}")
            
            # Display as Streamlit bar chart
            st.markdown("### 📈 Probability Distribution")
            st.bar_chart(prob_df.set_index('Class')['Probability'])
            
            # Expandable detailed table
            with st.expander("🔍 View All Probabilities"):
                prob_df['Probability_Display'] = prob_df['Probability'].apply(
                    lambda x: f"{x:.4f} ({x:.2%})"
                )
                st.dataframe(
                    prob_df[['Class', 'Probability_Display']], 
                    use_container_width=True
                )
        
        else:
            st.markdown("""
            <div class="info-card">
                <h4>👈 Ready for Classification</h4>
                <p>Upload an image on the left and click 'Classify Image' to see AI-powered results here.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <h4>🚀 Satellite Image Classification Platform</h4>
        <p>Powered by Advanced Deep Learning • Built with Streamlit</p>
        <p><em>Note: This demo uses simulated predictions. Deploy with TensorFlow for production use.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment suggestions with enhanced styling
    with st.expander("🚀 Deployment & Performance Tips"):
        st.markdown("""
        ### 🌟 **Recommended Deployment Platforms:**
        
        **🏆 Best for ML Models:**
        - **Hugging Face Spaces** - Optimized for ML workloads
        - **Google Cloud Run** - Serverless container deployment
        - **AWS SageMaker** - End-to-end ML platform
        
        **🔧 Performance Optimization:**
        - Convert models to TensorFlow Lite for 3x faster inference
        - Implement Redis caching for repeated predictions
        - Use GPU acceleration for real-time processing
        
        **📊 Model Enhancement:**
        - Fine-tune on domain-specific satellite imagery
        - Implement ensemble methods for higher accuracy
        - Add data augmentation for better generalization
        """)

if __name__ == "__main__":
    main()
