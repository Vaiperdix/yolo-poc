from ultralytics import YOLO
import json
import base64
from io import BytesIO
from PIL import Image

# class Pose:
def Predict(event):
    # get payload
    body = json.loads(event['body'])

    # get params
    img_b64 = body['image']
    version = body['version']
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
    return annotated_frame

# if __name__ == "__main__":
#     lambda_handler()