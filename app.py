from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return "HarvestAI - AI Farming Assistant"


@app.route("/crop-disease")
def crop_disease():
    return render_template("crop_disease.html")


if __name__ == "__main__":
    app.run(debug=True)
