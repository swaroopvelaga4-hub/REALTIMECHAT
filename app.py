s available", "Pulses"]
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


# ---------------- AI FARMING CHATBOT PAGE ----------------

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Please upload an image under 10 MB.", 413


# ---------------- RUN APP ----------------

# ---------------- CROP CALENDAR ----------------

@app.route("/crop-calendar")
def crop_calendar():
    return render_template("crop_calendar.html")


if __name__ == "__main__":
    app.run(debug=True)
