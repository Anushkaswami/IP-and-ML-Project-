# IP-and-ML-Project-
Blood Grouping system using image processing and machine learning 
🩸 Blood Grouping System Using Image Processing and Machine Learning
A deep learning-based system that predicts human blood group using thermal-style handprint images. This project leverages Convolutional Neural Networks (CNN) for image classification and integrates image processing techniques to improve accuracy. A Streamlit web application is also developed to allow users to interactively upload handprint images and get instant predictions.

🚀 Project Overview
The traditional method of determining blood groups involves serological testing with reagents and human intervention. In this project, we explore an automated approach using image processing and machine learning techniques to classify blood groups from thermal-style images of handprints, which visually represent features correlating with blood types.

Goal: Build a computer vision system that can accurately classify blood types such as A+, B+, AB+, O+, A−, B−, AB−, and O−.

✨ Key Features
🔍 Image preprocessing pipeline (grayscale, resizing, normalization)

🧠 CNN-based image classification

📊 Visualization of training and validation accuracy/loss

🌐 Streamlit web application for user-friendly interface

📁 Organized dataset structure for easy training

📂 Dataset
Source: Custom-generated or collected dataset of thermal-style handprint images.

Classes: A+, A−, B+, B−, AB+, AB−, O+, O−

Format: .jpg or .png images, grouped in folders per class.

You can upload or link your dataset in this section or refer to a public dataset (if available).

🧠 Model Architecture
Convolutional Neural Network (CNN) with:

3 Convolutional layers + MaxPooling

Flatten + Dense layers

Dropout layers to reduce overfitting

Activation: ReLU, Softmax

Optimizer: Adam

Loss Function: Categorical Crossentropy

🧰 Tools & Libraries Used
Python

TensorFlow / Keras

OpenCV – for image preprocessing

NumPy, Pandas – data handling

Matplotlib / Seaborn – visualization

Streamlit – to deploy web app interface

🖼️ Image Preprocessing
Each input image is:

Resized to 128x128 (or 224x224)

Converted to grayscale/RGB

Normalized to 0–1 scale

Augmented (optional) using:

Rotation

Flip

Zoom

📊 Training Performance
Model trained for ~25–30 epochs

Accuracy: ~90–95% on validation set

Loss curves and accuracy plots included

🌐 Streamlit Web App
Features:

Login page (basic authentication)

Upload button for thermal image

Real-time blood group prediction

Display confidence score

Run locally:

streamlit run app.py

🔧 Project Structure
css
Copy
Edit
blood-grouping-cnn/
│
├── dataset/
│   ├── A+/
│   ├── A-/ 
│   └── ... (8 folders total)
│
├── model/
│   └── blood_cnn_model.h5
│
├── notebooks/
│   └── model_training.ipynb
│
├── app.py  ← Streamlit App
├── requirements.txt
├── README.md
└── utils.py  ← Image preprocessing functions

💻 Installation & Setup


# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

📈 Future Work
Deploy on cloud (Heroku, HuggingFace Spaces, etc.)

