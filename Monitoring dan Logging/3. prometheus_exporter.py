from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import requests
import time
import sys

app = Flask(__name__)

REQUEST_TOTAL = Counter('app_requests_total', 'Total HTTP requests received')
SUCCESS_TOTAL = Counter('app_success_total', 'Total successful predictions')
FAIL_TOTAL = Counter('app_failed_total', 'Total failed predictions')
LATENCY = Histogram('app_latency_seconds', 'Time taken to process prediction')
CHURN_PRED = Counter('app_prediction_churn_total', 'Total Churn (1) predictions')
STAY_PRED = Counter('app_prediction_stay_total', 'Total Stay (0) predictions')
ACTIVE_REQ = Gauge('app_active_requests', 'Number of active requests being processed')
DATA_SIZE = Histogram('app_request_size_bytes', 'Size of incoming request payload')
MLFLOW_CALLS = Counter('app_mlflow_calls_total', 'Number of times MLflow model is called')
SYSTEM_UPTIME = Gauge('app_uptime_seconds', 'Uptime of the exporter service')

START_TIME = time.time()
MLFLOW_URL = "http://localhost:8080/invocations"

@app.route('/metrics')
def metrics():
    SYSTEM_UPTIME.set(time.time() - START_TIME)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQ.inc()
    REQUEST_TOTAL.inc()
    start_timer = time.time()
    
    try:
        data = request.json
        DATA_SIZE.observe(sys.getsizeof(str(data)))
        
        MLFLOW_CALLS.inc()
        response = requests.post(MLFLOW_URL, json=data)
        response.raise_for_status()
        
        prediction = response.json()
        
        result = prediction.get('predictions', [0])[0]
        
        if result == 1:
            CHURN_PRED.inc()
        else:
            STAY_PRED.inc()
            
        SUCCESS_TOTAL.inc()
        LATENCY.observe(time.time() - start_timer)
        ACTIVE_REQ.dec()
        
        return jsonify(prediction)

    except Exception as e:
        FAIL_TOTAL.inc()
        ACTIVE_REQ.dec()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Prometheus Exporter running in http://localhost:8000")
    app.run(host='0.0.0.0', port=8000)  