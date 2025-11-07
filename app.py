import os
os.environ["KERAS_BACKEND"] = "numpy"   # ✅ Run Keras without TensorFlow

import joblib
import numpy as np
import pandas as pd
from functools import wraps
from datetime import timedelta

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from keras.models import load_model

# =============== CONFIG ==================
APP_SECRET = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
ADMIN_USER = os.environ.get("APP_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("APP_ADMIN_PASS", "admin")

MODEL_PATH = "startup_success_ann.keras"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "label_encoders.joblib"
CSV_PATH = "startup data.csv"

# =============== APP ==================
app = Flask(__name__)
app.secret_key = APP_SECRET
app.permanent_session_lifetime = timedelta(hours=6)

# =============== LOAD MODEL + PREPROCESSORS ==================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print("[WARN] Could not load model:", e)

try:
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if not feature_names:
        raise Exception("Scaler missing feature names.")
except Exception as e:
    feature_names = []
    print("[WARN] Scaler load issue:", e)

try:
    encoders = joblib.load(ENCODERS_PATH)
except:
    encoders = {}

CATEGORY_ENCODER_KEY = "category_code"

# =============== AUTH HELPER ==================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

# =============== ROUTES ==================
@app.route("/")
def root():
    return redirect(url_for("login") if not session.get("logged_in") else url_for("index"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
            session["logged_in"] = True
            return redirect(url_for("index"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/index")
@login_required
def index():
    return render_template("index.html")


# ✅ FIX APPLIED HERE → REMOVED @login_required
@app.route("/api/prefill_row/<int:row_idx>")
def api_prefill_row(row_idx):
    try:
        df = pd.read_csv(CSV_PATH)
        if row_idx < 0 or row_idx >= len(df):
            return jsonify({"ok": False, "error": "Row out of range"}), 400
        row = preprocess_single_row(df.iloc[[row_idx]])
        data = {c: float(row.iloc[0][c]) for c in row.columns}
        return jsonify({"ok": True, "values": data, "features": feature_names})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def preprocess_single_row(row):
    row = row.copy()
    drop_cols = ['Unnamed: 6','state_code.1','zip_code','id','Unnamed: 0','object_id','state_code']
    for c in drop_cols:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    row['founded_at'] = pd.to_datetime(row['founded_at'], errors='ignore')
    row['closed_at']  = pd.to_datetime(row['closed_at'],  errors='ignore')

    ref = pd.to_datetime('2024-01-01')
    row['closed_at'] = (ref - row['closed_at']).dt.days.fillna(0).astype(int)

    if 'category_code' in row.columns and CATEGORY_ENCODER_KEY in encoders:
        row['category_code'] = encoders[CATEGORY_ENCODER_KEY].transform(row['category_code'].astype(str))

    drop2 = ['status','founded_at','first_funding_at','last_funding_at','name','city']
    for c in drop2:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_names]
    row = row.apply(pd.to_numeric, errors='coerce').fillna(0)

    return row


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    int_like = {"relationships","funding_rounds","funding_total_usd","milestones","is_top500",
                "founded_at_year","founded_at_day","founded_at_month","first_funding_at_year",
                "first_funding_at_day","first_funding_at_month","closed_at"}

    fields = feature_names
    has_category_text = CATEGORY_ENCODER_KEY in fields and CATEGORY_ENCODER_KEY in encoders

    if request.method == "POST":
        row = {}
        for f in fields:
            val = request.form.get(f, "").strip()
            row[f] = int(val) if f in int_like and val else float(val or 0)

        X = pd.DataFrame([row], columns=fields)
        X_sc = scaler.transform(X)
        prob = float(model.predict(X_sc, verbose=0).ravel()[0])
        result = "Acquired/Successful" if prob < 0.5 else "Closed/Not Acquired"
        flash(f"Prediction: {result} (prob={prob:.3f})", "success")

    return render_template("predict.html", fields=fields, has_category_text=has_category_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
