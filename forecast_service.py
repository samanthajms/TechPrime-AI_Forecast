from flask import Flask, request, jsonify
import inventory_forecasting as forecast
import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() with no arguments only looks for a file named exactly ".env" --
# it silently does NOT find "testing.env", which meant FORECAST_API_KEY was
# never actually loaded, and the auth check below was comparing against None
# (i.e. every request with no key passed, every request WITH the real key failed).
load_dotenv(Path(__file__).parent / 'testing.env')

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
def get_forecast_route():
    category = request.args.get('category', 'MEMORY')
    horizon = int(request.args.get('horizon', 3))

    try:
        results = forecast.get_forecast(category=category, horizon=horizon)
        return jsonify(results)
    except ValueError as e:
        # e.g. unknown category, or not enough history -- a client error, not a server crash
        return {'error': str(e)}, 400
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    # Run locally on port 5001
    app.run(host='127.0.0.1', port=5001, debug=True)