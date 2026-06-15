from flask import Flask, render_template, request
import numpy as np
import joblib
import cv2

app = Flask(__name__)

# Load trained model
model = joblib.load("Fingerprint based Diabetes detection.pkl")


# -------- Fingerprint Feature Extraction --------
def extract_features_from_image(file):
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (128, 128))

    mean = np.mean(img)
    std = np.std(img)

    edges = cv2.Canny(img, 50, 150)
    ridge_density = np.sum(edges) / (128 * 128)

    texture = np.var(img)

    return mean, std, ridge_density, texture


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- Get Fingerprint Image --------
        file = request.files["fingerprint"]
        fp_mean, fp_std, fp_ridge_density, fp_texture_var = extract_features_from_image(file)

        # -------- Clinical Inputs --------
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        bmi = float(request.form["bmi"])
        hba1c = float(request.form["hba1c"])
        glucose = float(request.form["glucose"])

        # -------- Categorical --------
        gender = request.form["gender"]
        smoking = request.form["smoking"]

        # -------- One-Hot Encoding --------
        gender_Male = 1 if gender == "Male" else 0

        smoking_current = 1 if smoking == "current" else 0
        smoking_ever = 1 if smoking == "ever" else 0
        smoking_former = 1 if smoking == "former" else 0
        smoking_never = 1 if smoking == "never" else 0
        smoking_not_current = 1 if smoking == "not current" else 0

        # -------- Final Input (MATCH MODEL ORDER) --------
        input_data = np.array([[ 
            fp_mean, fp_std, fp_ridge_density, fp_texture_var,
            age, hypertension, heart_disease, bmi,
            hba1c, glucose,
            gender_Male,
            smoking_current, smoking_ever, smoking_former,
            smoking_never, smoking_not_current
        ]])

        # -------- Prediction --------
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Diabetic ⚠️" if prediction == 1 else "Non-Diabetic ✅"

        return render_template(
            "index.html",
            prediction_text=result,
            confidence=f"{round(probability*100,2)}%"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error ❌",
            confidence=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)