import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
from joblib import load
# Load the trained model
loaded_model = load("final_model.h5")
# Initialize the InferenceHTTPClient with hardcoded API URL and API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Define a function to perform inference and return classes and probability scores
def perform_inference(image):
    try:
        custom_configuration = InferenceConfiguration(confidence_threshold=0.5)
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(image, model_id="kuchbhe/7")
        
        # Extract detected classes and their corresponding probability scores
        classes = [obj['class'] for obj in result['predictions']]
        scores = [obj['confidence'] for obj in result['predictions']]
        
        # Define a dictionary to map detected classes to their corresponding names
        class_names = {
            "AKIEC": "Actinic Keratosis",
            "BCC": "Basal Cell Carcinoma",
            "BKL": "Pigmented Benign Keratosis",
            "DF": "Dermatofibroma",
            "MEL": "Melanoma",
            "NV": "Nevus",
            "VASC": "Vascular Lesion"
        }
        
        # Replace detected classes with their corresponding names
        classes = [class_names.get(cls, cls) for cls in classes]
        
        return classes, scores
    except Exception as e:
        return [], []

def main():
    # Title
    st.title("Improved skin disease classification using generative adversarial network")
    
    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Perform inference when image is provided
    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Perform inference when button is clicked
        if st.button("Perform Inference"):
            with st.spinner("Performing inference..."):
                # Open the uploaded image
                image = Image.open(uploaded_image)
                
                # Perform inference
                classes, scores = perform_inference(image)
                
                # Display result with classes and probability scores
                if classes:
                    st.write("Detected Classes and Probability Scores:")
                    for cls, score in zip(classes, scores):
                        st.write(f"- {cls},Probability scores: {score:.2f}")
                else:
                    st.write("No classes detected.")

if __name__ == "__main__":
    main()