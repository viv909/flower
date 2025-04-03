import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
import time
from load import load_model  # Your existing load.py

# ---------------------------
# Set page configuration (must be first)
# ---------------------------
st.set_page_config(
    page_title="Flower Identifier", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS for Modern Glassmorphism Design
# ---------------------------
st.markdown(
    """
    <style>
    /* Base styling for the entire app */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* App background with subtle floral pattern */
    .stApp {
        background-image: linear-gradient(to bottom right, rgba(255, 255, 255, 0.8), rgba(240, 255, 240, 0.9)), 
                          url('https://www.transparenttextures.com/patterns/flowers.png');
        background-size: cover;
        background-attachment: fixed;
        padding: 0;
        margin: 0;
    }
    
    /* Main container with glassmorphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1100px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Card-based design for content sections */
    .content-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Typography enhancements */
    h1 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 20px !important;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    h2, h3, h4 {
        color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    
    p, label, div {
        color: #333 !important;
        font-weight: 400;
    }
    
    /* Enhanced file uploader */
    .stFileUploader {
        padding: 10px;
        border-radius: 15px;
        border: 2px dashed #6c5ce7;
        background: rgba(236, 240, 241, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #6c5ce7 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 8px 25px !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #5b4cce !important;
        box-shadow: 0 6px 15px rgba(108, 92, 231, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Enhanced radio buttons */
    .stRadio > div {
        display: flex;
        gap: 15px;
    }
    
    .stRadio label {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px 20px;
        border-radius: 30px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Image display enhancements */
    .img-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    
    /* Prediction details styling */
    .prediction-detail {
        padding: 8px 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .prediction-detail-label {
        font-weight: 600;
        color: #1e3a8a !important;
    }
    
    /* Custom badge */
    .badge {
        background: #6c5ce7;
        color: white !important;
        padding: 5px 12px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Success message */
    .success-msg {
        background-color: rgba(46, 213, 115, 0.2);
        color: #2ed573 !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2ed573;
        margin: 15px 0;
    }
    
    /* Error message */
    .error-msg {
        background-color: rgba(255, 71, 87, 0.2);
        color: #ff4757 !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4757;
        margin: 15px 0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            padding: 20px;
            margin: 10px;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        .content-card {
            padding: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Custom HTML Components
# ---------------------------
def custom_header():
    st.markdown(
        """
        <div style="text-align: center; padding: 0 0 20px 0;">
            <h1>Flower Identifier</h1>
            <p style="font-size: 1.2rem; color: #555 !important; max-width: 600px; margin: 0 auto;">
                Upload an image of a flower to identify its species, scientific details, and interesting facts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_card(content, title=None):
    card_html = f"<div class='content-card'>"
    if title:
        card_html += f"<h3>{title}</h3>"
    card_html += f"{content}</div>"
    
    return st.markdown(card_html, unsafe_allow_html=True)

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def load_flower_info(filename="flower.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    # Create a dictionary keyed by flower id
    return {flower["id"]: flower for flower in data}

flower_info = load_flower_info("flower.json")

@st.cache_resource
def get_model():
    return load_model("model.pth")

model = get_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_image(image):
    """Preprocess image, run inference, and return predicted flower info."""
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    info = flower_info.get(predicted_class, None)
    return predicted_class, info

def log_feedback(predicted_class, user_feedback, image_name=None):
    """Log feedback to a file for future model improvement."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback.log", "a") as f:
        f.write(f"[{timestamp}] Image: {image_name}, Predicted: {predicted_class}, Correction: {user_feedback}\n")

def show_loading_spinner():
    """Display a loading spinner."""
    with st.spinner("Analyzing your flower image..."):
        # Simulate processing time for better UX
        time.sleep(1.5)

def display_prediction_card(info):
    """Display prediction details in a card format."""
    if info is not None:
        details_html = f"""
        <div class='prediction-detail'>
            <span class='prediction-detail-label'>Flower Name:</span> {info['name'].title()}
        </div>
        <div class='prediction-detail'>
            <span class='prediction-detail-label'>Scientific Name:</span> <i>{info['scientific_name']}</i>
        </div>
        <div class='prediction-detail'>
            <span class='prediction-detail-label'>Genus:</span> {info['genus']}
        </div>
        <div class='prediction-detail'>
            <span class='prediction-detail-label'>Where Found:</span> {info['where_found']}
        </div>
        <div class='prediction-detail' style='border-bottom: none; padding-bottom: 15px;'>
            <span class='prediction-detail-label'>Fun Fact:</span> {info['fun_fact']}
        </div>
        <div>
            <span class='badge'>Identified with AI</span>
        </div>
        """
        st.markdown(details_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style='text-align: center; padding: 20px;'>
                <h4 style='color: #ff4757 !important;'>Unknown Flower</h4>
                <p>This flower is not in our database. Please provide feedback to help us improve!</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ---------------------------
# Main App Layout
# ---------------------------
# Wrap the entire content in our main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Display banner and header
banner_url = "flowers-identifier.webp"
st.image(banner_url, use_container_width=True)
custom_header()

# Upload section
upload_card_content = """
<p>Select a clear image of a flower for the most accurate identification.</p>
"""
create_card(upload_card_content, "Upload a Flower Image")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_name = uploaded_file.name
    
    # Create two columns for the image and prediction
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display the uploaded image in a container
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add image details below the image
        img_width, img_height = image.size
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 0.9rem; color: #666 !important;'>
                <p>Image size: {img_width} × {img_height} px | Filename: {image_name}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Show loading spinner while processing
        show_loading_spinner()
        
        # Get the prediction
        predicted_class, info = classify_image(image)
        
        # Show the prediction results in a card
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Identification Results")
        display_prediction_card(info)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feedback section in another card
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Was this identification correct?")
        
        # Add custom feedback buttons
        feedback = st.radio("", ("Yes", "No"), key="feedback_radio", horizontal=True)
        
        if feedback == "No":
            st.write("Please help us improve by providing the correct flower name:")
            user_correction = st.text_input("", key="user_correction", placeholder="Enter correct flower name...")
            
            if st.button("Submit Correction"):
                if user_correction.strip() != "":
                    log_feedback(predicted_class, user_correction.strip(), image_name)
                    st.markdown(
                        """
                        <div class='success-msg'>
                            <strong>Thank you for your feedback!</strong><br>
                            Your input helps us improve our model's accuracy.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class='error-msg'>
                            <strong>Please enter a valid flower name.</strong><br>
                            We need this information to improve our model.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            if st.button("Confirm"):
                log_feedback(predicted_class, "Correct", image_name)
                st.markdown(
                    """
                    <div class='success-msg'>
                        <strong>Thank you for confirming!</strong><br>
                        This helps us validate our model's performance.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        st.markdown("</div>", unsafe_allow_html=True)
        
else:
    # Display a placeholder or demo when no image is uploaded
    st.markdown(
        """
        <div style='text-align: center; padding: 40px 20px; background: rgba(255, 255, 255, 0.5); border-radius: 15px; margin-top: 20px;'>
            <img src="https://img.icons8.com/fluency/96/000000/flower.png" style="width: 80px; margin-bottom: 20px;">
            <h3>No Image Uploaded</h3>
            <p style="max-width: 500px; margin: 0 auto;">
                Upload a clear image of a flower to see its identification details. 
                Our AI model can identify hundreds of flower species.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add a footer
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0; margin-top: 30px; color: #888 !important; font-size: 0.8rem;'>
        <p>Flower Identifier • Powered by AI and Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Close the main container div
st.markdown("</div>", unsafe_allow_html=True)
