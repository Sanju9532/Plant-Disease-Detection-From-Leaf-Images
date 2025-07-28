#importing important libraries
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os
import matplotlib.pyplot as plt
import time

# App configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class names mapping
CLASS_NAMES = [
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Target Spot",
    "Tomato Spider Mites",
    "Tomato Septoria Leaf Spot",
    "Tomato Leaf Mold",
    "Tomato Late Blight",
    "Tomato Healthy",
    "Tomato Early Blight",
    "Tomato Bacterial Spot",
    "Potato Late Blight",
    "Potato Healthy",
    "Potato Early Blight",
    "Pepper Bell Healthy",
    "Pepper Bell Bacterial Spot"
]

# Load model function with caching
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app function
def main():
    st.title("🌿 Plant Disease Detection System")
    st.markdown("""
    Upload an image of a plant leaf to detect potential diseases. 
    This system can identify 15 different plant health conditions.
    """)

    # Sidebar for model upload and info
    with st.sidebar:
        st.header("Settings")
        model_file = st.file_uploader("Upload Model", type=['pkl'])
        
        if model_file:
            model_path = "temp_model.pkl"
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            model = load_model(model_path)
        else:
            model = None
        
        st.markdown("---")
        st.info("""
        **Model Requirements:**
        - Must be a .pkl file
        - Should be trained on the 15 disease classes
        - Expects input size of 150x150 pixels
        """)
        
        st.markdown("---")
        st.write("Developed by Sanjana.R")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploader"
        )

        if uploaded_file is not None:
            image_data = Image.open(uploaded_file)
            
            # Display uploaded image
            st.image(
                image_data,
                caption="Uploaded Image",
                use_container_width =True
            )

    with col2:
        st.subheader("Detection Results")
        
        if uploaded_file is not None and model is not None:
            if st.button("Run Detection", key="detect_button"):
                with st.spinner("Analyzing image..."):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    
                    try:
                        # Save temporarily to process
                        temp_path = "temp_img.jpg"
                        image_data.save(temp_path)
                        
                        # Preprocess and predict
                        img_array = preprocess_image(temp_path)
                        prediction = model.predict(img_array)
                        
                        # Get results
                        if prediction.ndim == 2 and prediction.shape[1] > 1:
                            pred_class = np.argmax(prediction, axis=1)[0]
                            confidence = np.max(prediction) * 100
                        else:
                            pred_class = int(prediction[0])
                            confidence = 100
                        
                        disease_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else "Unknown Disease"
                        
                        # Display results
                        st.markdown("---")
                        if "Healthy" in disease_name:
                            st.success(f"**Result:** {disease_name}")
                        else:
                            st.error(f"**Detected Disease:** {disease_name}")
                        
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        # Show probability distribution
                        if prediction.ndim == 2 and prediction.shape[1] > 1:
                            st.subheader("Probability Distribution")
                            fig, ax = plt.subplots()
                            ax.barh(CLASS_NAMES[:prediction.shape[1]], prediction[0] * 100)
                            ax.set_xlabel('Confidence (%)')
                            ax.set_title('Disease Probabilities')
                            st.pyplot(fig)
                        
                        # Add recommendation based on disease
                        st.markdown("---")
                        st.subheader("Recommendation")
                        if "Healthy" in disease_name:
                            st.success("Your plant appears healthy! Continue current care routine.")
                        else:
                            st.warning("Recommended actions:")
                            st.write("- Isolate affected plants")
                            st.write("- Remove severely infected leaves")
                            st.write("- Apply appropriate fungicide")
                            st.write("- Monitor plant recovery")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                    finally:
                        # Clean up temp files
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        progress_bar.empty()
        
        elif uploaded_file is None:
            st.info("Please upload an image on the left")
        elif model is None:
            st.warning("Please upload a model file in the sidebar")

if __name__ == "__main__":
    main()

