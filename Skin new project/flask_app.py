from flask import Flask, request, render_template
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
from joblib import load
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
#loaded_model = load("final_model.h5")

# Initialize the InferenceHTTPClient with hardcoded API URL and API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Define a function to perform inference
def perform_inference(image):
    try:
        custom_configuration = InferenceConfiguration(confidence_threshold=0.5)
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(image, model_id="kuchbhe/7")
        
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
        
        # Extract detected classes and probabilities
        classes = [class_names.get(obj['class'], obj['class']) for obj in result['predictions']]
        probabilities = [obj['confidence'] for obj in result['predictions']]
        
        # Combine classes and probabilities into a list of tuples
        predictions_with_prob = list(zip(classes, probabilities))
        
        return predictions_with_prob
    except Exception as e:
        return str(e)

    
@app.route("/")
def home():
    return render_template("home.html")    

# Define route for index page
@app.route("/upload", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded image
        uploaded_image = request.files["file"]
        
        # Open the uploaded image using PIL
        image = Image.open(uploaded_image)
        
        # Perform inference
        classes = perform_inference(image)
        
        # Render template with inference result
        return render_template("result.html", classes=classes)
    
    # Render index.html for GET request
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
