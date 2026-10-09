```python
from flask import Flask, render_template, request
import os
import requests
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Upload settings
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hugging Face token from Render Environment
HF_TOKEN = os.environ.get("HF_TOKEN")

# Crop disease AI model
DISEASE_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

# Soil classification AI model
SOIL_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "Ben041/soil-type-classifier"
)


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT PAGE ----------------

@app.route("/about")
def about():
    return "HarvestAI - AI Farming Assistant"


# ---------------- WEATHER PAGE ----------------

@app.route("/weather")
def weather():
    return render_template("weather.html")


# ---------------- CROP RECOMMENDATION PAGE ----------------

@app.route("/crop-recommendation")
def crop_recommendation():
    return render_template("crop_recommendation.html")


# ---------------- CROP DISEASE PAGE ----------------

@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


# ---------------- CROP DISEASE DETECTION ----------------

@app.route("/predict", methods=["POST"])
def predict():

    image = request.files.get("image")

    if not image or image.filename == "":
        return render_template(
            "crop_disease.html",
            result="Please select a crop or leaf image."
        )

    if not image.mimetype or not image.mimetype.startswith("image/"):
        return render_template(
            "crop_disease.html",
            result="Please upload a valid image."
        )

    filename = secure_filename(image.filename)

    if not filename:
        return render_template(
            "crop_disease.html",
            result="Invalid image filename."
        )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image.save(image_path)

    if not HF_TOKEN:
        return render_template(
            "crop_disease.html",
            result="Hugging Face API token is missing."
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    try:
        with open(image_path, "rb") as file:
            response = requests.post(
                DISEASE_MODEL_URL,
                headers=headers,
                data=file,
                timeout=60
            )

        print("DISEASE MODEL STATUS:", response.status_code)

        if response.status_code != 200:
            return render_template(
                "crop_disease.html",
                result="AI model could not analyze this image.",
                crop_name="Not detected",
                disease="AI analysis failed",
                treatment="Please try another clear crop image.",
                fertilizer="Not available",
                fertilizer_quantity="Not available",
                prevention="Please try again later.",
                image_url="/" + image_path
            )

        predictions = response.json()

        if not isinstance(predictions, list) or not predictions:
            return render_template(
                "crop_disease.html",
                result="No prediction was returned by the AI model.",
                crop_name="Not detected",
                disease="Unable to identify",
                treatment="Please try another clear image.",
                fertilizer="Not available",
                fertilizer_quantity="Not available",
                prevention="Consult a local agricultural expert.",
                image_url="/" + image_path
            )

        best = max(
            predictions,
            key=lambda item: item.get("score", 0)
        )

        label = best.get("label", "Unknown")
        score = best.get("score", 0)

        disease = f"{label} ({score * 100:.1f}% confidence)"

        return render_template(
            "crop_disease.html",
            result="Image analyzed successfully.",
            crop_name="See AI prediction",
            disease=disease,
            treatment="Confirm the diagnosis before choosing treatment.",
            fertilizer="Depends on the crop and confirmed diagnosis.",
            fertilizer_quantity="Get a recommendation from a local agricultural expert.",
            prevention="Use good field hygiene and monitor crop symptoms.",
            image_url="/" + image_path
        )

    except Exception as error:
        print("DISEASE DETECTION ERROR:", error)

        return render_template(
            "crop_disease.html",
            result="Something went wrong during AI analysis.",
            crop_name="Not detected",
            disease="Analysis failed",
            treatment="Please try again with a clear image.",
            fertilizer="Not available",
            fertilizer_quantity="Not available",
            prevention="Please try again later.",
            image_url="/" + image_path
        )


# ---------------- SEASON SUGGESTION ----------------

def get_season():
    month = datetime.now().month

    if month in [6, 7, 8, 9, 10]:
        return "Kharif (Rainy season)"

    elif month in [11, 12, 1, 2]:
        return "Rabi (Winter season)"

    else:
        return "Summer"

        
# ---------------- SOIL PHOTO DETECTION ----------------

@app.route("/soil-detect", methods=["POST"])
def soil_detect():

    image = request.files.get("soil_image")
    district = request.form.get("district", "").strip()

    if not image or image.filename == "":
        return render_template(
            "crop_recommendation.html",
            error="Please upload a soil photo."
        )

    if not image.mimetype or not image.mimetype.startswith("image/"):
        return render_template(
            "crop_recommendation.html",
            error="Please upload a valid image."
        )

    if not district:
        return render_template(
            "crop_recommendation.html",
            error="Please select your district."
        )

    if not HF_TOKEN:
        return render_template(
            "crop_recommendation.html",
            error="HF_TOKEN is missing in Render Environment."
        )

    try:
        image_bytes = image.read()

        if not image_bytes:
            return render_template(
                "crop_recommendation.html",
                error="The uploaded image is empty. Please try again."
            )

        response = requests.post(
            SOIL_MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            data=image_bytes,
            timeout=60
        )

        print("SOIL MODEL STATUS:", response.status_code)
        print("SOIL MODEL RESPONSE:", response.text[:500])

        if response.status_code != 200:
            return render_template(
                "crop_recommendation.html",
                error=(
                    "Soil AI service is unavailable or returned an error. "
                    "Please try again later."
                )
            )

        predictions = response.json()

        if not isinstance(predictions, list) or not predictions:
            return render_template(
                "crop_recommendation.html",
                error="The AI model did not return a soil prediction."
            )

        best = max(
            predictions,
            key=lambda item: item.get("score", 0)
        )

        soil_type = best.get("label", "Unknown")
        confidence = round(
            float(best.get("score", 0)) * 100,
            1
        )

        soil = soil_type.lower()

        # Preliminary crop suggestions only
        if "black" in soil:
            crops = ["Cotton", "Pulses", "Sorghum"]

        elif "red" in soil or "laterite" in soil:
            crops = ["Groundnut", "Millets", "Pulses"]

        elif "sand" in soil:
            crops = ["Groundnut", "Millets"]

        elif "loam" in soil or "alluvial" in soil:
            crops = [
                "Maize",
                "Pulses",
                "Rice if water is available"
            ]

        elif "clay" in soil:
            crops = [
                "Rice if water is available",
                "Pulses"
            ]

        else:
            crops = [
                "Soil type needs expert confirmation"
            ]

        season = get_season()

        return render_template(
            "crop_recommendation.html",
            soil_type=soil_type,
            confidence=confidence,
            district=district,
            season=season,
            crops=crops
        )

    except Exception as error:
        print("SOIL DETECTION ERROR:", error)

        return render_template(
            "crop_recommendation.html",
            error=(
                "Soil analysis failed. Try a clear photo "
                "of the soil in natural light."
            )
        )


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)
```
