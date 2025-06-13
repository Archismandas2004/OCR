import os
import uuid
from flask import Flask, render_template, request
from PIL import Image
import pytesseract

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd =r'C:\Users\Archi\OneDrive\Desktop\tesseract\tesseract.exe'  # Update this path as per your system

# Flask App Initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and text extraction."""
    extracted_text = None
    error = None

    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            error = "No file part in the request."
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected for upload."
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload an image file (png, jpg, jpeg, bmp)."
            else:
                # Save the file with a unique name
                unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)

                # Extract text from the image
                extracted_text = extract_text_from_image(file_path)

                # Optional: Remove the file after processing to save space
                os.remove(file_path)

    return render_template('upload_inline.html', text=extracted_text, error=error)

if __name__ == '__main__':
    app.run(debug=True)