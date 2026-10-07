from flask import Flask, render_template, request
import os
import requests

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_URL = "https://router.huggingface.co/hf-inference/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return "HarvestAI - AI Farming Assistant"


@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "crop_disease.html",
            result="Please upload a crop or leaf image."
        )

    image = request.files["image"]

    if image.filename == "":
        return render_template(
            "crop_disease.html",
            result="Please select an image."
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    with open(image_path, "rb") as f:
        response = requests.post(
            MODEL_URL,
            headers=headers,
            data=f
        )

    predictions = response.json()

    if isinstance(predictions, list) and len(predictions) > 0:

        best_prediction = predictions[0]

        label = best_prediction.get("label", "Unknown")
        score = best_prediction.get("score", 0)

        disease = f"{label} ({score * 100:.1f}% confidence)"

    else:
        disease = "Unable to identify the disease."

    return render_template(
        "crop_disease.html",
        result="Image analyzed successfully!",
        crop_name="Crop identification will appear after AI detection.",
        disease=disease,
        treatment="Treatment information will appear after AI detection.",
        fertilizer="Fertilizer recommendation will appear after AI detection.",
        fertilizer_quantity="Quantity will appear after AI detection.",
        prevention="Prevention advice will appear after AI detection.",
        image_url="/static/uploads/" + image.filename
    )


if __name__ == "__main__":
    app.run(debug=True)
