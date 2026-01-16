import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriVision: See & Spray Simulator", page_icon="🌱", layout="wide")

# --- HEADER & INTRODUCTION ---
st.title("🌱 Blue River Tech Simulation: See & Spray Logic")
st.markdown("""
**Candidate:** Sushma Mutyala | **Role:** Data Scientist 
**Objective:** Detect weeds, calculate spray area, and estimate herbicide savings.
""")

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("Simulation Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)  #whats this?
herbicide_cost = st.sidebar.number_input("Herbicide Cost ($/acre)", value=45.0)
field_size = st.sidebar.number_input("Field Size (Acres)", value=100)

# --- LOAD MODEL ---
# Using the standard YOLOv8 nano model for demonstration. 
# INSTRUCTION: In a real demo, you would load your custom-trained 'best.pt' here.
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Field Image (Corn/Weeds)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert file to image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run Inference
    results = model(image, conf=confidence_threshold)
    
    # --- PROCESS RESULTS FOR BUSINESS METRICS ---
    # In this demo, we simulate 'Weed' vs 'Crop' based on detected classes.
    # Standard YOLO detects 'potted plant' or 'vase'. 
    # NOTE: Your custom model would detect class '0: Weed' and '1: Crop'.
    
    detections = results[0].boxes
    weed_count = len(detections) # Assuming all detections are weeds for this generic demo
    
    # Visualize Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Computer Vision Analysis")
        # Plot the boxes on the image
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="YOLOv8 Detection", use_container_width=True)

    with col2:
        st.subheader("Agronomic Impact Report")
        
        # LOGIC: Calculate 'Spray Area' (Simulated)
        # If we found weeds, we spray. If not, we don't.
        # Let's assume each weed box represents a 2 sq ft spray zone.
        
        total_pixels = img_array.shape[0] * img_array.shape[1]
        weed_area_pixels = sum([box.xywh[0][2] * box.xywh[0][3] for box in detections])
        
        # Calculate Percentage of Field Sprayed
        spray_percentage = min((weed_area_pixels / total_pixels) * 100 * 5, 100) # x5 multiplier for visibility
        if spray_percentage < 1 and weed_count > 0: spray_percentage = 5.0 # Min floor
        
        # Financial Math
        total_cost_broadcast = field_size * herbicide_cost
        actual_cost_precision = (spray_percentage / 100) * total_cost_broadcast
        savings = total_cost_broadcast - actual_cost_precision
        
        # METRIC CARDS
        st.metric(label="Weeds Detected", value=f"{weed_count} instances")
        st.metric(label="Field Area Sprayed", value=f"{spray_percentage:.1f}%")
        
        st.success(f"💰 **Estimated Savings:** ${savings:,.2f} on {field_size} acres")
        
        # ERROR ANALYSIS (Matches JD Requirement)
        with st.expander("View Error Analysis / Confidence Levels"):
            st.write("Model Confidence for Detections:")
            for box in detections:
                st.write(f"Class: {int(box.cls)} | Conf: {float(box.conf):.4f}")

else:
    st.info("Please upload an image of crops/weeds to run the See & Spray simulation.")