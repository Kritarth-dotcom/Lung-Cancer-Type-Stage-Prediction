🫁 Lung Cancer Type & Stage Prediction

A deep learning-based web application that predicts lung cancer type and stage from chest CT scan images using a dual-head CNN architecture.

📌 Overview

This project uses a custom Dual ResNet-18 model to classify:

Cancer Type (6 canonical classes)
Cancer Stage (Stage 0 to Stage III/IV)

The application is deployed using Streamlit, allowing users to upload CT scan images and receive predictions in real-time.

🚀 Features
🧠 Deep Learning model based on PyTorch
🌐 Interactive UI using Streamlit
🔍 Automatic CT image validation (prevents random selfies from being classified as cancer… revolutionary concept)
📊 Dual prediction:
Cancer Type
Cancer Stage
📈 Probability distribution across canonical classes
⚡ Cached model loading for faster performance
🏗️ Model Architecture

The model is based on a modified ResNet-18:

Backbone: Pretrained ResNet-18 (ImageNet)
Two Output Heads:
Type Classification Head
Stage Classification Head

This allows simultaneous prediction of both cancer type and stage from the same feature representation.

📂 Project Structure
├── app.py                      # Streamlit application :contentReference[oaicite:3]{index=3}
├── report.ipynb               # Model training / analysis notebook
├── lung_cancer_stage_model_best.pth  # Trained model checkpoint
├── README.md                  # Project documentation
🧪 Supported Classes
Cancer Types (Canonical)
Normal
Benign Cases
Malignant Cases
Adenocarcinoma
Large Cell Carcinoma
Squamous Cell Carcinoma
Cancer Stages
Stage 0 (Normal)
Stage I
Stage II
Stage III/IV
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
2. Install dependencies
pip install -r requirements.txt
3. Add model checkpoint

Place your trained model file:

lung_cancer_stage_model_best.pth

in the root directory.

▶️ Running the App
streamlit run app.py

Then open your browser at:

http://localhost:8501
📸 How It Works
Upload a CT scan image
System validates if it's actually a CT scan (not your dog, not your face, not your assignment screenshot)
Model processes the image
Outputs:
Predicted cancer type
Predicted stage
Confidence score
Probability distribution
🧠 Key Components
Image Validation

A heuristic-based filter checks:

Contrast levels
Edge strength
Brightness distribution

This reduces false predictions on irrelevant images.

Prediction Pipeline
Image preprocessing (resize, normalize)
Feature extraction via ResNet backbone
Dual-head classification:
Type probabilities (softmax)
Stage probabilities (softmax)
Mapping to canonical classes
⚠️ Disclaimer

This project is:

❌ NOT for clinical use
❌ Not medically validated
✅ Intended for research and educational purposes only

If someone tries diagnosing actual patients with this, that’s not innovation, that’s a lawsuit.

📊 Tech Stack
Python
PyTorch
Streamlit
NumPy
PIL (Image Processing)
🔮 Future Improvements
Replace heuristic CT validation with a trained classifier
Add Grad-CAM for model explainability
Support 3D CT volumes instead of single slices
Deploy using cloud services (AWS/GCP)
