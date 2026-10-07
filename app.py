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

    return render_template(
        "crop_disease.html",
        result="Image uploaded successfully!",
        crop_name="Crop identification will appear after AI detection.",
        disease="AI disease detection will be connected next.",
        treatment="Treatment information will appear after AI detection.",
        fertilizer="Fertilizer recommendation will appear after AI detection.",
        fertilizer_quantity="Quantity will appear after AI detection.",
        prevention="Prevention advice will appear after AI detection.",
        image_url="/static/uploads/" + image.filename
    )


if __name__ == "__main__":
    app.run(debug=True)
