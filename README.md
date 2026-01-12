🧠 Brain Tumor Detection Using Convolutional Neural Networks

This project implements an end-to-end deep learning system for brain tumor detection from MRI images using a Convolutional Neural Network (CNN).
The system covers the full lifecycle: data preprocessing, model training, evaluation, interpretability, and web-based deployment.

📌 Project Overview

Brain tumor detection is a critical medical imaging task. This project aims to:

Automatically classify brain MRI images as Tumor or Healthy

Demonstrate strong generalization performance

Provide model interpretability through feature map visualization

Deploy the trained model as an interactive web application using Hugging Face Spaces

🗂 Dataset

Type: Brain MRI images

Classes:

Tumor

Healthy

Total Images: ~4600

Image formats: JPG, PNG, TIFF

Preprocessing handled: RGB/Grayscale conversion, resizing, normalization

Dataset Split

Training: 80%

Validation: 10%

Test: 10%

Splitting is performed automatically using split-folders.

🛠 Technologies Used

Python 3

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Google Colab

Gradio

Hugging Face Spaces

🧠 Model Architecture

The CNN architecture consists of:

4 Convolutional layers with ReLU activation

MaxPooling layers for spatial reduction

Fully connected Dense layers

Sigmoid output for binary classification

Input size: 224 × 224 × 3
Total parameters: ~9.6 million

🔄 Training Details

Optimizer: RMSprop

Learning rate: 1e-4

Loss function: Binary Crossentropy

Epochs: 40

Batch size: 32

Data augmentation:

Rotation

Horizontal flip

📊 Evaluation Results
Accuracy & Loss

Training accuracy converges to ~100%

Validation accuracy stabilizes at 97–98%

Loss curves show stable convergence with no overfitting

Confusion Matrix

High true positive and true negative rates

Very low false negatives (important for medical context)

Classification Report

Accuracy: 98%

Precision: 0.98

Recall: 0.97–0.98

F1-score: 0.98

These results confirm robust and balanced performance.

🔍 Model Interpretability

To better understand the internal behavior of the CNN:

Feature maps were extracted from convolutional layers

Early layers capture edges and textures

Deeper layers focus on abstract, tumor-relevant regions

This analysis improves transparency and trust in the model’s decisions.

🌐 Web Deployment (Hugging Face Spaces) : https://huggingface.co/spaces/mehigh77777/CNN_Brain_tumor

The trained model is deployed as an interactive web application using Gradio.

Features

Upload MRI images directly from browser

Real-time prediction

Probability scores for both classes

Automatic preprocessing at inference time

Interface

Input: MRI image (PNG/JPG)

Output: Tumor / Healthy with confidence scores
