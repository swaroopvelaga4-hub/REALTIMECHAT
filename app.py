
from flask import Flask, render_template, request
import os
import requests
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- UPLOAD SETTINGS ----------------

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------- AI MODEL URLS ----------------

DISEASE_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

SOIL_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "Ben041/soil-type-classifier"
)

# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return "HarvestAI - AI Farming Assistant"


# ---------------- WEATHER ----------------

@app.route("/weather")
def weather():
    return render_template("weather.html")


# ---------------- CROP RECOMMENDATION ----------------

@app.route("/crop-recommendation")
def crop_recommendation():
    return render_template("crop_recommendation.html")


# ---------------- SOIL DOCTOR PAGE ----------------

@app.route("/soil-doctor")
def soil_doctor():
    return render_template("soil_detection.html")


# ---------------- CROP DISEASE PAGE ----------------

@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


# ---------------- CROP DISEASE DETECTION ----------------

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get("image")

    if not image or not image.filename:
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

    filename = (
        datetime.now().strftime("%Y%m%d%H%M%S%f")
        + "_" + filename
    )

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)
    image_url = "/static/uploads/" + filename

    if not HF_TOKEN:
        return render_template(
            "crop_disease.html",
            result="Hugging Face API token is missing.",
            image_url=image_url
        )

    try:
        with open(image_path, "rb") as file:
            response = requests.post(
                DISEASE_MODEL_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                data=file,
                timeout=60
            )

        if response.status_code != 200:
            print("DISEASE MODEL STATUS:", response.status_code)
            return render_template(
                "crop_disease.html",
                result="AI model unavailable. Please try again later.",
                crop_name="Not detected",
                disease="Analysis unavailable",
                treatment="Confirm the diagnosis with an expert.",
                fertilizer="Not available",
                fertilizer_quantity="Not available",
                prevention="Consult a local agricultural expert.",
                image_url=image_url
            )

        predictions = response.json()

        if not isinstance(predictions, list) or not predictions:
            return render_template(
                "crop_disease.html",
                result="No prediction returned by the AI model.",
                image_url=image_url
            )

        best = max(
            predictions,
            key=lambda item: item.get("score", 0)
        )

        label = best.get("label", "Unknown")
        confidence = float(best.get("score", 0)) * 100

        return render_template(
            "crop_disease.html",
            result="AI prediction received. Please verify the result.",
            crop_name="See predicted label",
            disease=f"{label} ({confidence:.1f}% confidence)",
            treatment="Confirm the diagnosis with an agricultural expert.",
            fertilizer="Depends on the confirmed crop and disease.",
            fertilizer_quantity="Follow soil-test-based advice.",
            prevention="Monitor plants and maintain field hygiene.",
            image_url=image_url
        )

    except requests.RequestException as error:
        print("DISEASE API ERROR:", error)
        return render_template(
            "crop_disease.html",
            result="Could not connect to the AI service. Please retry.",
            image_url=image_url
        )

    except Exception as error:
        print("DISEASE DETECTION ERROR:", error)
        return render_template(
            "crop_disease.html",
            result="An error occurred during image analysis.",
            image_url=image_url
        )


# ---------------- SEASON SUGGESTION ----------------

def get_season():
    month = datetime.now().month

    if month in [6, 7, 8, 9, 10]:
        return "Kharif (Rainy season)"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi (Winter season)"
    else:
        return "Summer"


# ---------------- PEST DOCTOR PAGE ----------------

@app.route("/pest-doctor")
def pest_doctor():
    return render_template("pest_identification.html")


# ---------------- PEST IDENTIFICATION ----------------

@app.route("/pest-identify", methods=["POST"])
def pest_identify():
    pest_image = request.files.get("pest_image")
    crop = request.form.get("crop", "").strip()

    if not pest_image or not pest_image.filename:
        return render_template(
            "pest_identification.html",
            error="Please upload a pest photo."
        )

    if not pest_image.mimetype or not pest_image.mimetype.startswith("image/"):
        return render_template(
            "pest_identification.html",
            error="Please upload a valid image."
        )

    if not crop:
        return render_template(
            "pest_identification.html",
            error="Please select a crop."
        )

    filename = secure_filename(pest_image.filename)

    if not filename:
        return render_template(
            "pest_identification.html",
            error="Invalid image filename."
        )

    filename = (
        datetime.now().strftime("%Y%m%d%H%M%S%f")
        + "_" + filename
    )

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    pest_image.save(image_path)

    # Pest AI model is not connected yet.
    return render_template(
        "pest_identification.html",
        error=(
            f"{crop} photo uploaded successfully. "
            "AI pest detection is not connected yet."
        ),
        image_url="/static/uploads/" + filename
    )


# ---------------- SOIL PHOTO DETECTION ----------------

@app.route("/soil-detect", methods=["POST"])
def soil_detect():
    image = request.files.get("soil_image")
    district = request.form.get("district", "").strip()

    if not image or not image.filename:
        return render_template(
            "soil_detection.html",
            error="Please upload a soil photo."
        )

    if not image.mimetype or not image.mimetype.startswith("image/"):
        return render_template(
            "soil_detection.html",
            error="Please upload a valid image."
        )

    if not district:
        return render_template(
            "soil_detection.html",
            error="Please select your district."
        )

    if not HF_TOKEN:
        return render_template(
            "soil_detection.html",
            error="HF_TOKEN is missing in Render Environment."
        )

    filename = secure_filename(image.filename)

    if not filename:
        return render_template(
            "soil_detection.html",
            error="Invalid image filename."
        )

    filename = (
        datetime.now().strftime("%Y%m%d%H%M%S%f")
        + "_" + filename
    )

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image_bytes = image.read()

        if not image_bytes:
            return render_template(
                "soil_detection.html",
                error="The uploaded image is empty."
            )

        with open(image_path, "wb") as saved_image:
            saved_image.write(image_bytes)

        response = requests.post(
            SOIL_MODEL_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image_bytes,
            timeout=60
        )

        print("SOIL MODEL STATUS:", response.status_code)

        if response.status_code != 200:
            return render_template(
                "soil_detection.html",
                error="Soil AI service unavailable. Please try again."
            )

        predictions = response.json()

        if not isinstance(predictions, list) or not predictions:
            return render_template(
                "soil_detection.html",
                error="The AI model did not return a soil prediction."
            )

        best = max(
            predictions,
            key=lambda item: item.get("score", 0)
        )

        soil_type = best.get("label", "Unknown")
        confidence = round(float(best.get("score", 0)) * 100, 1)
        soil = soil_type.lower()

        if "black" in soil:
            crops = ["Cotton", "Pulses", "Sorghum"]
        elif "red" in soil or "laterite" in soil:
            crops = ["Groundnut", "Millets", "Pulses"]
        elif "sand" in soil:
            crops = ["Groundnut", "Millets"]
        elif "loam" in soil or "alluvial" in soil:
            crops = ["Maize", "Pulses", "Rice if water is available"]
        elif "clay" in soil:
            crops = ["Rice if water is available", "Pulses"]
        else:
            crops = ["Confirm soil type before selecting a crop"]

        return render_template(
            "soil_detection.html",
            soil_type=soil_type,
            confidence=confidence,
            district=district,
            season=get_season(),
            crops=crops
        )

    except requests.RequestException as error:
        print("SOIL API ERROR:", error)
        return render_template(
            "soil_detection.html",
            error="Could not connect to the soil AI service. Please retry."
        )

    except Exception as error:
        print("SOIL DETECTION ERROR:", error)
        return render_template(
            "soil_detection.html",
            error="Soil analysis failed. Please try another clear photo."
        )


# ---------------- MARKET PRICES ----------------

@app.route("/market-prices", methods=["GET", "POST"])
def market_prices():
    if request.method == "POST":
        crop = request.form.get("crop", "").strip()
        state = request.form.get("state", "").strip()
        district = request.form.get("district", "").strip()
        market = request.form.get("market", "").strip()

        if not crop or not state or not district:
            return render_template(
                "market_prices.html",
                error="Please select a crop, state, and district."
            )

        # Live market data source is not connected yet.
        return render_template(
            "market_prices.html",
            error=(
                "Live market prices are not connected yet. "
                "No current prices are available to display. "
                "A verified market data source is required."
            ),
            selected_crop=crop,
            selected_state=state,
            selected_district=district,
            selected_market=market
        )

    return render_template("market_prices.html")


# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Please upload an image under 10 MB.", 413


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)
