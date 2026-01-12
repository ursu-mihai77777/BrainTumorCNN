import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os


IMG_SIZE = 224
MODEL_PATH = "BrainTumors.keras"
CLASS_NAMES = {
    0: "Tumor",
    1: "Healthy"
}

# --- 2. LOAD MODEL ---
print("Loading model...")
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: The file '{MODEL_PATH}' was not found in the repository.")

        model = tf.keras.Sequential()
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    model = None

# --- 3. PREPROCESSING ---
def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
        # Convert to numpy array
    img_array = np.array(image)
    

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    # Normalize pixel values (0-1)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 4. PREDICTION FUNCTION ---
def predict_tumor(image):
    if model is None:
        return {"Error": "Model not loaded"}

    img = preprocess_image(image)
    prob = model.predict(img)[0][0]

    # prob = probabilitatea clasei 1 (Healthy)
    if prob >= 0.5:
        return {
            "Healthy": float(prob),
            "Tumor": float(1 - prob)
        }
    else:
        return {
            "Tumor": float(1 - prob),
            "Healthy": float(prob)
        }




interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload MRI Scan"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🧠 Brain Tumor Detection",
    description="Upload a Brain MRI image to detect potential tumors.",
    examples=[] 
)

# --- 6. LAUNCH ---
if __name__ == "__main__":

    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False, 
        debug=False
    )
