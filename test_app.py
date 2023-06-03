import pytest
import requests

def test_server_running():
    response = requests.get('http://localhost:5002')
    assert response.status_code == 200

def test_predict_route():
    data = {'data': [8.3252, 41.0, 6.984, 1.02, 322.0, 2.55, 37.88, -122.23]}  # example data
    response = requests.post('http://localhost:5002/predict', json=data)
    assert response.status_code == 200
    assert 'prediction' in response.json()