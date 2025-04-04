# 🌺 Good Morning: Flower Identification System

An AI-powered web application that identifies flowers from uploaded images and provides rich botanical information including name, scientific details, genus, fun facts, and geographical distribution — all through an elegant, professional user interface.

---

## 📸 Features

- 🌸 **Image Upload**: Upload a flower image and get instant identification.
- 🧠 **AI Model Prediction**: Uses a pre-trained deep learning model (PyTorch) to classify flowers accurately.
- 🧾 **Botanical Details**: Displays common name, scientific name, genus, fun fact, and found-in region.
- 💬 **Feedback System**: Users can submit corrections to improve the model over time.
- 💻 **Responsive UI**: Built with Streamlit and styled with glassmorphism for a clean, modern look.
- 🔄 **CI/CD Integrated**: GitHub Actions for testing and validation.
- 🧪 **Unit Tested**: Includes tests for core components such as prediction and feedback logging.

---

## 🛠️ Technologies Used

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Model Inference**: [PyTorch](https://pytorch.org/)
- **Image Processing**: [Pillow](https://python-pillow.org/)
- **CI/CD**: GitHub Actions
- **Testing**: `unittest` (Python standard library)
- **Data Storage**: JSON for flower data and plain text for feedback logs
- **model.pth**:https://huggingface.co/AventIQ-AI/Resnet-50-flower-classification

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
-▶️ Run the App and 🧪 Running Tests
  ```bash
  streamlit run app1.py
  python -m unittest test_app.py

