from flask import Flask, request, jsonify
import asyncio

from ai.captionConversion import generate_instagram_caption
from ai.generateImageCaption import generateImageCaption

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'


@app.route('/generate-caption', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type or no file selected"}), 400

    result = generateImageCaption(file)
    caption, error = result['caption'], result['error']
    if error:
        return jsonify({"error": error}), 500

    converted_caption = generate_instagram_caption(caption)

    return jsonify({'caption': converted_caption})
