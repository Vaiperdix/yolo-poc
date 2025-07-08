from flask import Flask, request, jsonify
import prediction  # Tu script de predicción

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Recibir JSON del frontend
    data = request.json
    
    # Llamar a tu función de predicción
    results = prediction.Predict(data)
    
    # Retornar array de floats
    return jsonify({"predictions": results.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)