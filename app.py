from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import joblib, traceback
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Rutas a los pipelines
PIPE_DESMP = os.path.join('resources', 'pipeline_desempeno.pkl')
PIPE_DESER = os.path.join('resources', 'pipeline_desercion.pkl')

# Cargar los pipelines
pipe_desempeno = joblib.load(PIPE_DESMP)
pipe_desercion = joblib.load(PIPE_DESER)

# Ruta de prueba
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API activa'})

# Ruta para predecir DESEMPEÑO DEL ALUMNO
@app.route('/predict_desempeno', methods=['POST'])
@cross_origin(origin='http://localhost:5173')
def predict_desempeno():
    try:
        data = request.get_json(force=True)
        X_input = pd.DataFrame([data])
        pred = pipe_desempeno.predict(X_input)[0]
        label_map = {0: 'RENDIMIENTO BAJO', 1: 'RENDIMIENTO ALTO'}
        return jsonify({'prediction': int(pred), 'interpretation': label_map[pred]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta para predecir SE DIO DE BAJA
@app.route('/predict_desercion', methods=['POST'])
@cross_origin(origin='http://localhost:5173')
def predict_desercion():
    try:
        data = request.get_json(force=True)
        X_input = pd.DataFrame([data])
        pred = pipe_desercion.predict(X_input)[0]
        label_map = {0: 'PERMANECE', 1: 'BAJA'}
        return jsonify({'prediction': int(pred), 'interpretation': label_map[pred]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
