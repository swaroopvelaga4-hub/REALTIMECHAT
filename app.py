from flask import Flask, render_template, request, jsonify
import os
import requests
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- SETTINGS ----------------

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

DISEASE_MODEL = (
    "https://router.huggingface.co/hf-inference/models/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

SOIL_MODEL = (
    "https://router.huggingface.co/hf-inference/models/"
    "Ben041/soil-type-classifier"
)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


# ---------------- HELPERS ----------------

def get_season():
    month = datetime.now().month

    if month in [6, 7, 8, 9]:
        return "Kharif"
    elif month in [10, 11, 12, 1]:
        return "Rabi"
    return "Summer"


def save_uploaded_image(field_name):
    image = request.files.get(field_name)

    if not image or not image.filename:
        return None, "Please select an image."

    filename = secure_filename(image.filename)

    if not filename:
        return None, "Invalid filename."

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        return None, "Upload a JPG, JPEG, PNG, or WEBP image."

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{timestamp}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    image.save(path)
    return (image, filename, path), None


def hf_image_prediction(model_url, image_file):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing from Render environment variables.")

    image_file.stream.seek(0)

    response = requests.post(
        model_url,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": image_file.mimetype or "application/octet-stream"
        },
        data=image_file.read(),
        timeout=60
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"AI service returned HTTP {response.status_code}. Please retry later."
        )

    result = response.json()

    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))

    if not isinstance(result, list) or not result:
        raise RuntimeError("The AI service returned no prediction.")

    return sorted(
        result,
        key=lambda item: float(item.get("score", 0)),
        reverse=True
    )


def money_value(value):
    try:
        amount = float(value)
        if amount < 0:
            return None
        return amount
    except (TypeError, ValueError):
        return None


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- WEATHER ----------------

@app.route("/weather", methods=["GET", "POST"])
def weather():
    result = None
    error = None

    if request.method == "POST":
        location = request.form.get("location", "").strip()

        if not location:
            error = "Please enter a town or district."
        else:
            try:
                geo_response = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "language": "en"},
                    timeout=15
                )
                geo_response.raise_for_status()
                places = geo_response.json().get("results", [])

                if not places:
                    error = "Location not found. Check the spelling and try again."
                else:
                    place = places[0]
                    weather_response = requests.get(
                        "https://api.open-meteo.com/v1/forecast",
                        params={
                            "latitude": place["latitude"],
                            "longitude": place["longitude"],
                            "current": (
                                "temperature_2m,relative_humidity_2m,"
                                "precipitation,wind_speed_10m"
                            ),
                            "timezone": "auto",
                            "forecast_days": 1
                        },
                        timeout=15
                    )
                    weather_response.raise_for_status()
                    current = weather_response.json().get("current", {})

                    result = {
                        "location": (
                            place.get("name", location)
                            + ", "
                            + place.get("admin1", place.get("country", ""))
                        ),
                        "temperature": current.get("temperature_2m", "N/A"),
                        "humidity": current.get("relative_humidity_2m", "N/A"),
                        "rainfall": current.get("precipitation", "N/A"),
                        "wind": current.get("wind_speed_10m", "N/A"),
                        "time": current.get("time", "")
                    }

            except requests.RequestException:
                error = "Weather service is unavailable. Please try again later."
            except (ValueError, KeyError, TypeError):
                error = "Could not read weather information. Please retry."

    return render_template("weather.html", result=result, error=error)


# ---------------- CROP RECOMMENDATION ----------------

@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    crops = None
    error = None

    if request.method == "POST":
        soil = request.form.get("soil", "").strip().lower()
        season = request.form.get("season", get_season()).strip().lower()
        water = request.form.get("water", "").strip().lower()

        if not soil:
            error = "Please select or enter your soil type."
        else:
            recommendations = {
                "black": ["Cotton", "Pulses", "Sorghum"],
                "red": ["Groundnut", "Millets", "Pulses"],
                "sandy": ["Groundnut", "Millets"],
                "loam": ["Maize", "Pulses", "Vegetables"],
                "alluvial": ["Rice", "Maize", "Pulses"],
                "clay": ["Rice", "Pulses"]
            }

            crops = recommendations.get(soil)

            if crops is None:
                error = (
                    "Soil type not recognised. Please select black, red, "
                    "sandy, loam, or alluvial soil."
                )
            elif water in ["low", "limited", "no"]:
                crops = [
                    crop for crop in crops
                    if crop.lower() not in ["rice"]
                ]

    return render_template(
        "crop_recommendation.html",
        crops=crops,
        error=error,
        season=get_season()
    )


# ---------------- CROP DISEASE / LEAF DOCTOR ----------------

@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded, error = save_uploaded_image("image")

    if error:
        # Support the existing page without requiring a new template.
        return render_template("crop_disease.html", error=error)

    image_file, filename, path = uploaded

    try:
        predictions = hf_image_prediction(DISEASE_MODEL, image_file)
        best = predictions[0]
        label = str(best.get("label", "Unknown"))
        confidence = round(float(best.get("score", 0)) * 100, 2)

        return render_template(
            "crop_disease.html",
            prediction=label,
            disease=label,
            confidence=confidence,
            image_url=f"/static/uploads/{filename}",
            predictions=predictions[:3],
            note=(
                "AI result is a preliminary suggestion, not a confirmed diagnosis. "
                "Check with a local agricultural expert before applying chemicals."
            )
        )

    except Exception as exc:
        app.logger.exception("Disease detection failed: %s", exc)
        return render_template(
            "crop_disease.html",
            error=(
                "AI disease detection is temporarily unavailable. "
                "Please try a clear photo again later."
            )
        )


# ---------------- PEST DOCTOR ----------------

@app.route("/pest-doctor")
def pest_doctor():
    return render_template("pest_identification.html")


@app.route("/pest-identify", methods=["POST"])
def pest_identify():
    uploaded, error = save_uploaded_image("pest_image")

    if error:
        return render_template("pest_identification.html", error=error)

    image_file, filename, path = uploaded

    # Pest AI model is not configured yet. Do not invent a pest diagnosis.
    return render_template(
        "pest_identification.html",
        error=(
            "Image uploaded successfully, but pest AI identification is not "
            "connected yet. Please consult an agricultural expert for "
            "identification and treatment advice."
        ),
        image_url=f"/static/uploads/{filename}"
    )


# ---------------- SOIL DOCTOR ----------------

@app.route("/soil-doctor")
def soil_doctor():
    return render_template("soil_detection.html")


@app.route("/soil-detect", methods=["POST"])
def soil_detect():
    uploaded, error = save_uploaded_image("soil_image")
    district = request.form.get("district", "").strip()

    if error:
        return render_template(
            "soil_detection.html",
            error=error,
            district=district
        )

    image_file, filename, path = uploaded

    try:
        predictions = hf_image_prediction(SOIL_MODEL, image_file)
        best = predictions[0]
        soil_type = str(best.get("label", "Unknown"))
        confidence = round(float(best.get("score", 0)) * 100, 2)

        soil_key = soil_type.lower()
        if "black" in soil_key:
            crops = ["Cotton", "Pulses", "Sorghum"]
        elif "red" in soil_key or "laterite" in soil_key:
            crops = ["Groundnut", "Millets", "Pulses"]
        elif "sand" in soil_key:
            crops = ["Groundnut", "Millets"]
        elif "loam" in soil_key or "alluvial" in soil_key:
            crops = ["Maize", "Pulses", "Rice where water is sufficient"]
        elif "clay" in soil_key:
            crops = ["Rice where water is sufficient", "Pulses"]
        else:
            crops = ["Confirm soil type with a local soil-testing service."]

        return render_template(
            "soil_detection.html",
            soil_type=soil_type,
            confidence=confidence,
            district=district,
            season=get_season(),
            crops=crops,
            image_url=f"/static/uploads/{filename}",
            note=(
                "Photo-based soil classification is preliminary. "
                "For accurate soil fertility and fertilizer advice, get a soil test."
            )
        )

    except Exception as exc:
        app.logger.exception("Soil detection failed: %s", exc)
        return render_template(
            "soil_detection.html",
            error="Soil AI service is unavailable. Please retry later.",
            district=district
        )


# ---------------- MARKET PRICES ----------------

@app.route("/market-prices", methods=["GET", "POST"])
def market_prices():
    if request.method == "GET":
        return render_template("market_prices.html")

    crop = request.form.get("crop", "").strip()
    state = request.form.get("state", "").strip()
    district = request.form.get("district", "").strip()
    market = request.form.get("market", "").strip()

    if not crop or not state or not district:
        return render_template(
            "market_prices.html",
            error="Please select a crop, state, and district.",
            selected_crop=crop,
            selected_state=state,
            selected_district=district,
            selected_market=market
        )

    # No verified AGMARKNET API key/resource configuration is present yet.
    return render_template(
        "market_prices.html",
        error=(
            "Live market prices are not connected yet. No current prices "
            "are available to display. A verified government market-data "
            "API and its configuration are required."
        ),
        selected_crop=crop,
        selected_state=state,
        selected_district=district,
        selected_market=market
    )


# ---------------- AI FARMING CHATBOT PAGE ----------------

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    reply = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            error = "Please type a farming question."
        else:
            # Safe fallback; a real AI chatbot requires a configured AI service.
            reply = (
                "I can help with general farming information. For a reliable "
                "answer about a specific crop disease, pesticide, or fertilizer "
                "dose, consult your local agricultural officer. AI chatbot "
                "service is not connected yet."
            )

    return render_template("chatbot.html", reply=reply, error=error)


# ---------------- CROP CALENDAR ----------------

@app.route("/crop-calendar")
def crop_calendar():
    return render_template("crop_calendar.html", season=get_season())


# ---------------- FARM EXPENSE & PROFIT TRACKER ----------------

@app.route("/farm-tracker")
def farm_tracker():
    return render_template("farm_tracker.html")


# ---------------- SIMPLE HEALTH CHECK ----------------

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "HarvestAI",
        "time": datetime.utcnow().isoformat() + "Z"
    })


# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Please upload an image under 10 MB.", 413


@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "index.html",
        error="The requested page was not found."
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Internal server error")
    return "Something went wrong. Please try again later.", 500


if __name__ == "__main__":
    app.run(debug=False)
