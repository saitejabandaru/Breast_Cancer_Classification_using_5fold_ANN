import os
import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)

MEAN_FEATURES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]

SE_FEATURES = ["radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"]

WORST_FEATURES = ["radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

FEATURE_NAMES = MEAN_FEATURES + SE_FEATURES + WORST_FEATURES

MODEL_DIR = "fold_models"

models = []
scalers = []

for i in range(1, 6):
    model_path = os.path.join(MODEL_DIR, f"ann_fold_{i}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_fold_{i}.pkl")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    models.append(model)
    scalers.append(scaler)


def predict_with_ensemble(input_features):
    input_array = np.array(input_features, dtype=float).reshape(1, -1)

    all_preds = []

    for model, scaler in zip(models, scalers):
        scaled_input = scaler.transform(input_array)
        pred = model.predict(scaled_input, verbose=0)[0][0]
        all_preds.append(pred)

    avg_pred = np.mean(all_preds)

    # Training mapping:
    # M = 1, B = 0
    if avg_pred >= 0.5:
        return "Malignant"
    else:
        return "Benign"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        name = request.form.get("name", "User")

        input_features = []

        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            input_features.append(float(value))

        prediction = predict_with_ensemble(input_features)

        if prediction == "Benign":
            return render_template("success.html", name=name)
        else:
            return render_template("failure.html", name=name)

    return render_template(
        "predict.html",
        mean_features=MEAN_FEATURES,
        se_features=SE_FEATURES,
        worst_features=WORST_FEATURES
    )


if __name__ == "__main__":
    app.run(debug=True)