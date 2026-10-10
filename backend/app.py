
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/style.css")
def styles():
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "applicant_income",
            "coapplicant_income",
            "loan_amount",
            "loan_term",
            "credit_score",
            "employment_type",
            "existing_loans"
        ]

        if not data or any(field not in data for field in required_fields):
            return jsonify({
                "error": "Please provide all applicant details."
            }), 400

        # Prepare applicant details for the model
        applicant = pd.DataFrame([{
            "applicant_income": float(data["applicant_income"]),
            "coapplicant_income": float(data["coapplicant_income"]),
            "loan_amount": float(data["loan_amount"]),
            "loan_term": int(data["loan_term"]),
            "credit_score": int(data["credit_score"]),
            "employment_type": str(data["employment_type"]),
            "existing_loans": int(data["existing_loans"])
        }])

        # Validate input values
        if (
            applicant.select_dtypes(include="number").isna().any().any()
            or not applicant.select_dtypes(include="number").map(
                lambda value: value >= 0
            ).all().all()
        ):
            return jsonify({
                "error": "Numeric values must be valid and non-negative."
            }), 400

        if applicant.loc[0, "employment_type"] not in [
            "Salaried", "Self-employed"
        ]:
            return jsonify({
                "error": "Invalid employment type."
            }), 400

        if not 300 <= applicant.loc[0, "credit_score"] <= 900:
            return jsonify({
                "error": "Credit score must be between 300 and 900."
            }), 400

        prediction = model.predict(applicant)[0]

        return jsonify({
            "prediction": prediction,
            "message": (
                "This is an educational prediction from synthetic data, "
                "not an actual bank loan decision."
            )
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please enter valid numeric values."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to generate a prediction."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
