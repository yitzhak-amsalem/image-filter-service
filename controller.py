from flask import Flask, request, jsonify
from flask_cors import CORS
from request import Request
from face_detection import process

app = Flask(__name__)
CORS(app, allow_headers='Content-Type')


@app.route('/filter', methods=['POST'])
def uploadImage():
    try:
        request_data = request.get_json()
        request_model = Request(**request_data)
        filter_images = process(request_model)
        return jsonify({"result": filter_images})
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500


if __name__ == '__main__':
    app.run()
