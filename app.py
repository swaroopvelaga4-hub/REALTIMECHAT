from flask import Flask, render_template, request
import os
app = Flask(__name__)

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

    try:
        ai_result = analyze(image_path, include_gradcam=False)

        if not ai_result["is_mango"]:
            return render_template(
                "crop_disease.html",
                result="This image could not be confidently identified as a mango leaf."
            )

        disease = ai_result["predicted_class"]
        confidence = round(ai_result["confidence"] * 100, 2)

        disease_info = ai_result.get("disease_info") or {}

        treatment = " ".join(
            disease_info.get("remedies", [])
        )

        fertilizer = "Fertilizer recommendation will be added using verified agricultural guidance."

        fertilizer_quantity = (
            "Use fertilizer quantity only according to "
            "soil test results and local agricultural recommendations."
        )

        prevention = " ".join(
            disease_info.get("symptoms", [])
        )

        return render_template(
            "crop_disease.html",
            result="AI analysis completed!",
            disease=f"{disease} ({confidence}% confidence)",
            treatment=treatment,
            fertilizer=fertilizer,
            fertilizer_quantity=fertilizer_quantity,
            prevention=prevention,
            image_url="/static/uploads/" + image.filename
        )

    except Exception as e:
        print("AI ERROR:", e)

        return render_template(
            "crop_disease.html",
            result="AI analysis could not be completed. Please try again."
        )


if __name__ == "__main__":
    app.run(debug=True)
