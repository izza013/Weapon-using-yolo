import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# Function to load YOLOv10 model
@st.cache_resource
def load_model(weights_path):
    checkpoint = torch.load(weights_path, map_location=torch.device('cuda'))  # Load model on GPU
    model = checkpoint['model'].float()  # Convert model to single precision
    model = model.to(torch.device('cuda'))  # Ensure the model is on GPU
    model.eval()  # Set model to evaluation mode
    return model

# Load the model (replace with the path to your model weights file)
model = load_model('/content/drive/MyDrive/weaponFinal/yolov10_best.pt')

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image, (640, 640))  # Resize to YOLOv10 input size
    image_resized = np.transpose(image_resized, (2, 0, 1))  # Change to (C, H, W)
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_resized = torch.tensor(image_resized, dtype=torch.float32) / 255.0  # Normalize
    image_resized = image_resized.to(torch.device('cuda'))  # Send image tensor to GPU
    return image_resized

# Function to draw bounding boxes
def draw_boxes(image, predictions):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for detection in predictions:
        for *xyxy, conf, cls in detection:
            xyxy = list(map(int, xyxy))
            conf = float(conf)
            cls = int(cls)
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {cls} Conf {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Streamlit app
st.title("Weapon Detection using YOLOv10")



uploaded_file = st.file_uploader("Upload an image of a weapon", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
      
    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Run inference
    with torch.no_grad():
        results = model(input_tensor)[0]  # Perform inference
        predictions = results.cpu().numpy()  # Convert results to numpy array

    # Draw bounding boxes and display results
    image_result = draw_boxes(image, predictions)
    st.image(image_result, caption="Detections", use_column_width=True)
