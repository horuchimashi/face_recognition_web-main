import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from datetime import datetime
from facedb import FaceDB

# Initialize Flask app and FaceDB instance
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key. Note: This is less critical for AJAX flash messages.

# Initialize FaceDB
# Make sure the 'facedata' directory exists or is created by facedb
db = FaceDB(
    path="facedata",
    metric="euclidean",
    embedding_dim=128,
    module="face_recognition",
)

# Function to save image with a timestamp (optional for debugging/logging, not strictly needed for facedb)
def save_image_with_timestamp(img_bytes, folder="uploads"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as f:
        f.write(img_bytes)
    return file_path

@app.route("/", methods=["GET", "POST"])
def add_face_data():
    if request.method == "POST":
        try:
            name = request.form.get("name", "").strip()
            img_file = request.files.get("image")

            if not name:
                return jsonify({"message": "Name is required!", "category": "danger"}), 400

            if not img_file or img_file.filename == "":
                return jsonify({"message": "No file selected!", "category": "danger"}), 400

            img_bytes = img_file.read()
            if not img_bytes:
                return jsonify({"message": "Uploaded file is empty!", "category": "danger"}), 400

            face_id = db.add(name, img=img_bytes)
            return jsonify({"message": f"Face added successfully with ID: {face_id}", "category": "success"}), 200
        except ValueError as e:
            if "No face detected" in str(e):
                return jsonify({"message": "No face detected in the uploaded image.", "category": "warning"}), 400
            return jsonify({"message": str(e), "category": "danger"}), 500
        except Exception as e:
            return jsonify({"message": f"An unexpected error occurred: {str(e)}", "category": "danger"}), 500
    return render_template("index.html")

@app.route("/recognize", methods=["GET", "POST"])
def recognize_face():
    if request.method == "POST":
        img_file = request.files.get("image")

        if not img_file or img_file.filename == "":
            return jsonify({"message": "No file selected for recognition!", "category": "danger"}), 400

        img_bytes = img_file.read()
        if not img_bytes:
            return jsonify({"message": "Uploaded file is empty!", "category": "danger"}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"message": "Could not decode image. Invalid image format?", "category": "danger"}), 400

        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = db.recognize(img=rgb_img, include=["name", "confidence"])

            # Corrected: Access attributes directly from the FaceResults object
            if result and result.id: # Check if a face ID was found (i.e., a match)
                return jsonify({
                    "name": result.name,
                    "confidence": f"{result.confidence:.2f}%",
                    "message": f"Recognized as {result.name} ({result.confidence:.2f}%)",
                    "category": "success"
                }), 200
            else:
                # This block will be hit if result is None or result.id is None/empty
                return jsonify({"name": None, "message": "No known face recognized.", "category": "warning"}), 200
        except ValueError as e:
            if "No face detected" in str(e):
                return jsonify({"name": None, "message": "No face detected in the uploaded image for recognition.", "category": "warning"}), 400
            return jsonify({"message": str(e), "category": "danger"}), 500
        except Exception as e:
            return jsonify({"message": f"An unexpected error occurred during recognition: {str(e)}", "category": "danger"}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)