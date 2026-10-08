from flask import Flask, render_template, request
import os
import requests
from werkzeug.utils import secure_filename

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


@app.route("/weather")
def weather():
    return render_template("weather.html")


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

    filename = secure_filename(image.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image.save(image_path)

    # Check Hugging Face token
    if not HF_TOKEN:
        return render_template(
            "crop_disease.html",
            result="Hugging Face API token is missing."
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    try:

        with open(image_path, "rb") as f:

            response = requests.post(
                MODEL_URL,
                headers=headers,
                data=f,
                timeout=60
            )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        if response.status_code != 200:

            return render_template(
                "crop_disease.html",
                result="AI model could not analyze the image.",
                crop_name="Not detected",
                disease="AI analysis failed",
                treatment="Please try another clear crop or leaf image.",
                fertilizer="Not available",
                fertilizer_quantity="Not available",
                prevention="Please upload a clear image.",
                image_url="/static/uploads/" + filename
            )

        predictions = response.json()

        if isinstance(predictions, list) and len(predictions) > 0:

            best_prediction = predictions[0]

            label = best_prediction.get(
                "label",
                "Unknown"
            )

            score = best_prediction.get(
                "score",
                0
            )

            disease = f"{label} ({score * 100:.1f}% confidence)"

        else:

            disease = "Unable to identify the disease."

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "crop_disease.html",
            result="Something went wrong during AI analysis.",
            crop_name="Not detected",
            disease="Analysis failed",
            treatment="Please try again with a clear image.",
            fertilizer="Not available",
            fertilizer_quantity="Not available",
            prevention="Please try again.",
            image_url="/static/uploads/" + filename
        )

    return render_template(
        "crop_disease.html",
        result="Image analyzed successfully!",
        crop_name="AI crop identification coming next.",
        disease=disease,
        treatment="Treatment information will be added next.",
        fertilizer="Fertilizer recommendation will be added next.",
        fertilizer_quantity="Quantity recommendation will be added next.",
        prevention="Prevention advice will be added next.",
        image_url="/static/uploads/" + filename
    )


if __name__ == "__main__":
    app.run(debug=True)
