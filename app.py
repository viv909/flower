import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
from load import load_model  # Your existing load.py

# ---------------------------
# Set page configuration (must be first)
# ---------------------------
st.set_page_config(page_title="Flower Identifier", layout="wide")

# ---------------------------
# Custom CSS for the Look
# ---------------------------
# This CSS targets Streamlit's root container (.stApp) and forces a white background.
# It also defines a main container with our custom styling.
st.markdown(
    """
    <style>
    /* Force the entire app to have a white background */
    .stApp {
        background-color: #ffffff;
    }
    /* Create a main container for content */
    .main-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 40px;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }
    /* Set text color to black */
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #000000 !important;
    }

    /* Hide the default Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

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

def log_feedback(predicted_class, user_feedback):
    """Log feedback to a file for future model improvement."""
    with open("feedback.log", "a") as f:
        f.write(f"Predicted: {predicted_class}, Correction: {user_feedback}\n")

# ---------------------------
# Layout: Banner, Title, and Main Container
# ---------------------------

# Display a banner image at the top
banner_url = "flowers-identifier.webp"
st.image(banner_url, use_container_width=True)

# Wrap our main content in a custom container div
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.title("Flower Classification")
st.write("Upload a flower image to identify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image and prediction side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        predicted_class, info = classify_image(image)
        st.header("Prediction")
        if info is not None:
            st.markdown(f"**Flower Name:** {info['name'].title()}")
            st.markdown(f"**Scientific Name:** {info['scientific_name']}")
            st.markdown(f"**Genus:** {info['genus']}")
            st.markdown(f"**Fun Fact:** {info['fun_fact']}")
            st.markdown(f"**Where Found:** {info['where_found']}")
        else:
            st.markdown("**Prediction:** This flower is not in our database.")
        
        st.markdown("---")
        st.subheader("Is this prediction correct?")
        feedback = st.radio("", ("Yes", "No"), key="feedback_radio")
        
        if feedback == "No":
            st.write("Please enter the correct flower name:")
            user_correction = st.text_input("", key="user_correction")
            if st.button("Submit Correction"):
                if user_correction.strip() != "":
                    log_feedback(predicted_class, user_correction.strip())
                    st.success("Thank you for your feedback! We'll use it to improve our model.")
                else:
                    st.error("Please enter a valid correction.")

st.markdown("</div>", unsafe_allow_html=True)

