import os
os.environ["KERAS_BACKEND"] = "numpy"   # ✅ Run Keras without TensorFlow

import joblib
import numpy as np
import pandas as pd
from functools import wraps
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from keras.models import load_model   # ✅ Pure Keras load_model

# =========================
# Config
# =========================
APP_SECRET = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
ADMIN_USER = os.environ.get("APP_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("APP_ADMIN_PASS", "admin")

MODEL_PATH = "startup_success_ann.keras"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "label_encoders.joblib"
CSV_PATH = "startup data.csv"  # ✅ Must be uploaded to repo

# =========================
# App
# =========================
app = Flask(__name__)
app.secret_key = APP_SECRET
app.permanent_session_lifetime = timedelta(hours=6)

# =========================
# Load Model + Scaler + Encoders
# =========================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
except Exception as e:
    feature_names = []
    print(f"[WARN] Could not load scaler or infer feature names: {e}")

try:
    encoders = joblib.load(ENCODERS_PATH)
except:
    encoders = {}

CATEGORY_ENCODER_KEY = "category_code"


# =========================
# Auth Decorator
# =========================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap


# =========================
# Routes
# =========================
@app.route("/")
def root():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
            session["logged_in"] = True
            return redirect(url_for("index"))
        flash("Invalid Credentials", "danger")
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


# ✅ FIX: REMOVE login_required so JS can access it
@app.route("/_features")
def _features():
    return jsonify({"feature_names": feature_names})


# =========================
# Preprocessing Helper
# =========================
def preprocess_row(raw_row):
    row = raw_row.copy()

    # Drop unused
    drop_cols = ['Unnamed: 6','state_code.1','zip_code','id','Unnamed: 0','object_id','state_code']
    for c in drop_cols:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    # Date handling
    row['founded_at'] = pd.to_datetime(row['founded_at'], errors='coerce')
    row['closed_at']  = pd.to_datetime(row['closed_at'], errors='coerce')
    row['first_funding_at'] = pd.to_datetime(row['first_funding_at'], errors='coerce')
    row['last_funding_at']  = pd.to_datetime(row['last_funding_at'], errors='coerce')

    # Validation rule
    if (row['closed_at'] - row['founded_at']).iloc[0] < pd.Timedelta(0):
        raise ValueError("closed_at occurs before founded_at.")

    # Convert closed_at to days offset
    ref = pd.to_datetime("2024-01-01")
    row['closed_at'] = (ref - row['closed_at']).dt.days.fillna(0).astype(int)

    # Fill milestone NaNs
    fill_zero = ['age_first_milestone_year','age_last_milestone_year']
    for c in fill_zero:
        if c in row.columns:
            row[c] = row[c].fillna(0)

    # Encode category
    if 'category_code' in row.columns and CATEGORY_ENCODER_KEY in encoders:
        row['category_code'] = encoders[CATEGORY_ENCODER_KEY].transform(row['category_code'].astype(str))

    # Drop fields not used in model
    drop2 = ['status','founded_at','first_funding_at','last_funding_at','name','city']
    for c in drop2:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    # Ensure all required columns
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    return row[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)


# =========================
# Predict UI
# =========================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    fields = feature_names[:]
    has_category_text = CATEGORY_ENCODER_KEY in fields and CATEGORY_ENCODER_KEY in encoders

    if request.method == "POST":
        row = {}
        for f in fields:
            val = request.form.get(f, "").strip()
            row[f] = float(val) if val else 0

        X = pd.DataFrame([row], columns=fields)
        X_sc = scaler.transform(X)
        prob = float(model.predict(X_sc, verbose=0).ravel()[0])
        pred = int(prob >= 0.5)
        label = {0: "Acquired / Successful", 1: "Closed / Failed"}[pred]

        return render_template("predict.html", fields=fields, prob=prob, pred=label)

    return render_template("predict.html", fields=fields, has_category_text=has_category_text)


# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
