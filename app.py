from settings.constants import TRAIN_CSV, VAL_CSV
from utils import DataLoader, Predictor
from flask import Flask, request, jsonify, make_response
import pandas as pd
import json

from utils.model_search.model_search import ModelSearcher

app = Flask(__name__)


def upload(path):
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        uploaded_file.save(path)
        return make_response('Uploaded', 200)

    else:
        return make_response('Error', 400)


@app.route('/upload_val', methods=['POST'])
def upload_val():
    upload(VAL_CSV)


@app.route('/upload_val', methods=['POST'])
def upload_train():
    upload(TRAIN_CSV)


@app.route('/fit', methods=['GET'])
def fit():
    searcher = ModelSearcher()
    searcher.fit()
    summary = searcher.models.to_dict()
    return make_response(jsonify(summary), 200)


@app.route('/predict', methods=['GET'])
def predict():
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)

    data = json.loads(request.form.get(received_keys[0]))
    df = pd.DataFrame.from_dict(data)

    loader = DataLoader()
    loader.fit(df)
    processed_df = loader.load_data()

    predictor = Predictor()
    response_dict = {'prediction': predictor.predict(processed_df).tolist()}

    return make_response(jsonify(response_dict), 200)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
