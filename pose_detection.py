import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-pose.pt")

    
    cap = cv2.VideoCapture(0) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        results = model(frame)
        
        for result in results:
            print("\nInformación de la detección:")
            print(f"Número de personas detectadas: {len(result.boxes)}")
            if len(result.boxes) > 0:
                print("\nDetalles de la primera persona:")
                print(f"Coordenadas del bounding box: {result.boxes[0].xyxy}")
                print(f"Puntos clave detectados: {result.keypoints[0].xy}")
                print(f"Confianza de la detección: {result.boxes[0].conf}")
        
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLO Pose Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 