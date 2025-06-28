from flask import Flask, request, jsonify
from car_ai_module import CarImageProcessor
import uuid

app = Flask(__name__)
processor = CarImageProcessor()

@app.route("/report", methods=["POST"])
def report_car():
    try:
        data = request.get_json()

        if not data or 'user_id' not in data or 'image_data' not in data or 'incident_type' not in data:
            return jsonify({"error": "Missing required fields"}), 400

        user_id = data['user_id']
        image_data = data['image_data']
        incident_type = data['incident_type']
        report_data = data.get('report_data', {})

        img_array = processor.preprocess_image_from_user(image_data)
        if img_array is None:
            return jsonify({"error": "Image processing failed"}), 400

        car_region = processor.detect_car_in_user_image(img_array)
        features = processor.extract_features_from_user_image(car_region)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 400

        report_id = str(uuid.uuid4())
        image_url = "https://example.com/fake.jpg"  # مؤقتًا

        response = {
            "success": True,
            "report_id": report_id,
            "image_url": image_url,
            "matches_found": 0,
            "message": "تم رفع البلاغ بنجاح"
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
