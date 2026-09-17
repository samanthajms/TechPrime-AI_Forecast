from flask import Flask, request, jsonify
import inventory_forecasting as forecast
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Security check before hitting the DB or models
@app.before_request
def check_api_key():
    if request.endpoint != 'health': # Exclude health check from auth
        if request.headers.get('X-API-Key') != os.environ.get('FORECAST_API_KEY'):
            return {'error': 'unauthorized'}, 401

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    category = request.args.get('category', 'MEMORY')
    horizon = int(request.args.get('horizon', 3))
    
    try:
        # Calls the function which now queries your local DB
        results = forecast.generate_forecast(category, horizon) 
        return jsonify(results)
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '_main_':
    # Run locally on port 5001
    app.run(host='127.0.0.1', port=5001, debug=True)