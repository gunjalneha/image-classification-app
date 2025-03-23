from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from FLASK"


from classifier import classify_image

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    img_path = f"./uploads/{file.filename}"
    file.save(img_path)
    prediction = classify_image(img_path)
    return prediction