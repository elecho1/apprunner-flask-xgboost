from flask import Flask, jsonify, request
import numpy as np
import pickle
import xgboost as xgb

app = Flask(__name__)
MODEL_PATH = "./artifact/xgboost_model.pickle"

def load_artifact(filename):
    artifact = None
    with open(filename, mode='rb') as fp:
        artifact = pickle.load(fp)
    if artifact is None:
        raise ValueError        
    return artifact

model = load_artifact(MODEL_PATH)


@app.route("/")
def index():
    return "XGBoost prediction API with App Runner and flask."


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    """
    こちらのサイトを参考にしています。https://qiita.com/fam_taro/items/1464c42324f15d7b8223
    """
    global model

    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if request.get_json().get("feature"):
        feature = request.get_json().get("feature")

        feature = np.array(feature)
        if len(feature) != 2:
            feature = feature.reshape((1, -1))
        
        dfeature = xgb.DMatrix(feature)
        response["prediction"] = model.predict(dfeature).tolist()

        response["success"] = True

    return jsonify(response)


if __name__ == "__main__":
    app.run()
