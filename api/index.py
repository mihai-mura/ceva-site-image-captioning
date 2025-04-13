from flask import Flask, request, jsonify
from flask_cors import CORS
import io, requests

from ai.captionConversion import generate_instagram_caption
from ai.generateImageCaption import generateImageCaption

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/generate-caption', methods=['POST'])
def upload_file():
    data = request.get_json()
    if 'imageUrl' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Download the image from the URL
        response = requests.get(data['imageUrl'])
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a file-like object from the downloaded content
        file = io.BytesIO(response.content)

        result = generateImageCaption(file)
        caption, error = result['caption'], result['error']
        if error:
            return jsonify({"error": error}), 500

        converted_caption = generate_instagram_caption(caption)
        return jsonify({'caption': converted_caption})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()