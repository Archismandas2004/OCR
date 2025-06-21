import os
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request
from PIL import Image
import pytesseract
from openai import OpenAI

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Archi\OneDrive\Desktop\tesseract\tesseract.exe'

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "PUT_OPENAI_KEY_HERE"

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flask App Initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# YOLO paths
LABELS_PATH = r"D:\c\programs\greendukan\objectdetect\yolo-coco\coco.names"
WEIGHTS_PATH = r"D:\c\programs\greendukan\objectdetect\yolo-coco\yolov3.weights"
CONFIG_PATH = r"D:\c\programs\greendukan\objectdetect\yolo-coco\yolov3.cfg"

LABELS = open(LABELS_PATH).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

def detect_objects_in_image(image_path, confidence_threshold=0.5, nms_threshold=0.3):
    """Perform object detection on the image using YOLO."""
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
    
    # Determine output layer names
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Construct blob and forward pass
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Loop through detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    result_text = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            label = LABELS[classIDs[i]]
            confidence = confidences[i]
            result_text.append(f"{label}: {confidence:.2f}")
    return result_text if result_text else ["No objects detected."]

def get_material_scores_and_alternatives(brand_name, object_name, materials):
    """Fetch toxicity, recyclability scores, and eco-friendly alternatives."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that provides material analysis and eco-friendly suggestions."},
                {"role": "user", "content": f"The brand is '{brand_name}', the object is '{object_name}', and the detected materials are:\n\n{materials}.\n\nProvide toxicity, recyclability scores, and eco-friendly alternatives."}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching material scores and alternatives: {e}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload, text extraction, and object detection."""
    brand_name = None
    object_name = None
    extracted_text = None
    detected_objects = None
    material_scores_and_alternatives = None
    error = None

    if request.method == 'POST':
        brand_name = request.form.get('brand_name', '')
        object_name = request.form.get('object_name', '')

        if 'file' not in request.files:
            error = "No file part in the request."
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected for upload."
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload an image file (png, jpg, jpeg, bmp)."
            else:
                unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)

                # Perform OCR and object detection
                extracted_text = extract_text_from_image(file_path)
                detected_objects = detect_objects_in_image(file_path)

                # Combine materials and fetch scores
                materials = f"Extracted Text: {extracted_text}\nDetected Objects: {', '.join(detected_objects)}"
                material_scores_and_alternatives = get_material_scores_and_alternatives(brand_name, object_name, materials)

                os.remove(file_path)

    return render_template(
        'upload_inline.html',
        brand_name=brand_name,
        object_name=object_name,
        text=extracted_text,
        objects=detected_objects,
        material_scores_and_alternatives=material_scores_and_alternatives,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
