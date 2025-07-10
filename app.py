from flask import Flask, request, jsonify
from ultralytics import YOLO
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    try:
        print("Starting prediction...")
        # Recibir JSON del frontend
        data = request.json
        
        # get payload
        body = json.loads(data['body'])

        # get params
        img_b64 = body['image']
        version = body['version']
        print("version: ", version)

        # open image
        img = Image.open(BytesIO(base64.b64decode(img_b64.encode('ascii'))))

        model_map = {
            "n": "./models/yolo11n-pose.pt",
            "s": "./models/yolo11s-pose.pt",
            "m": "./models/yolo11m-pose.pt",
            "l": "./models/yolo11l-pose.pt",
            "x": "./models/yolo11x-pose.pt"
        }

        model = YOLO(model_map.get(version, "./models/yolo11n-pose.pt"))
        
        results = model(img)
        
        for result in results:
            print("\nInformación de la detección:")
            print(f"Número de personas detectadas: {len(result.boxes)}")
            if len(result.boxes) > 0:
                print("\nDetalles de la primera persona:")
                print(f"Coordenadas del bounding box: {result.boxes[0].xyxy}")
                print(f"Puntos clave detectados: {result.keypoints[0].xy}")
                print(f"Confianza de la detección: {result.boxes[0].conf}")
        
        annotated_frame = results[0].plot()
        print("Results: ", annotated_frame)
        print("Ending prediction...")
        return annotated_frame
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()