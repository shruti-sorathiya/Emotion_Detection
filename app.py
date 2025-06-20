from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# DFS-based decision tree for emotion classification
def dfs_emotion_classification(node, features):
    if "result" in node:
        print(f"Final Emotion: {node['result']}")  # Debugging output
        return node["result"]

    condition = node["condition"]
    if condition(features):
        print(f"Condition met: {condition} -> TRUE branch")  # Debugging output
        return dfs_emotion_classification(node["true"], features)
    else:
        print(f"Condition not met: {condition} -> FALSE branch")  # Debugging output
        return dfs_emotion_classification(node["false"], features)



# Decision tree structure
def get_emotion_tree():
    return {
        "condition": lambda features: features["mouth"] == "upward",
        "true": {
            "condition": lambda features: features["eyebrows"] == "furrowed",
            "true": {"result": "neutral"},      #neutral
            "false": {"result": "happy"},       #happy
        },
        "false": {
            "condition": lambda features: features["eyebrows"] == "furrowed",
            "true": {"result": "sad"},       # Sad: Downward mouth + furrowed eyebrows
            "false": {"result": "angry"},    #angry
        },
    }



# Extract facial features
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return {"mouth": "unknown", "eyebrows": "unknown"}

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Detect mouth state
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        mouth_state = "upward" if len(smiles) > 0 else "downward"

        # Detect eyebrow state using eye positions
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        if len(eyes) >= 2:
            eye_y_positions = [ey[1] for ey in eyes]
            eyebrow_state = "furrowed" if max(eye_y_positions) - min(eye_y_positions) > 10 else "relaxed"
        else:
            eyebrow_state = "unknown"

        return {"mouth": mouth_state, "eyebrows": eyebrow_state}

    return {"mouth": "unknown", "eyebrows": "unknown"}



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    features = extract_features(filepath)
    emotion_tree = get_emotion_tree()
    detected_emotion = dfs_emotion_classification(emotion_tree, features)

    return render_template('result.html', emotion=detected_emotion, image_url=url_for('static', filename=f"uploads/{filename}"))


if __name__ == '__main__':
    app.run(debug=True)
