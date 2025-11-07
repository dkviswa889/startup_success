import os
os.environ["KERAS_BACKEND"] = "numpy"   # ✅ Use Keras Core (no TensorFlow)

import joblib
import numpy as np
import pandas as pd
from functools import wraps
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from keras.models import load_model


# =========================
# Config
# =========================
APP_SECRET = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
ADMIN_USER = os.environ.get("APP_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("APP_ADMIN_PASS", "admin")

MODEL_PATH = "startup_success_ann.keras"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "label_encoders.joblib"
CSV_PATH = "startup_data.csv"


# =========================
# App Setup
# =========================
app = Flask(__name__)
app.secret_key = APP_SECRET
app.permanent_session_lifetime = timedelta(hours=6)


# =========================
# Load ML Assets
# =========================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
except:
    feature_names = []

try:
    encoders = joblib.load(ENCODERS_PATH)
except:
    encoders = {}

CATEGORY_ENCODER_KEY = "category_code"


# =========================
# Auth Helper
# =========================
def login_required(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrap


# =========================
# Routes (UI Pages)
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
        flash("Invalid username or password", "danger")
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


# =========================
# Data for Frontend (No Auth Needed)
# =========================
@app.route("/_features")
def _features():
    return jsonify({"feature_names": feature_names})


# =========================
# LOAD & FILL CSV API  ✅ FIXED (no login_required here)
# =========================
@app.route("/api/prefill_row/<int:row_idx>")
def api_prefill_row(row_idx):
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        return jsonify({"ok": False, "error": f"CSV read failed: {e}"}), 500

    if row_idx < 0 or row_idx >= len(df):
        return jsonify({"ok": False, "error": "Row index out of range"}), 404

    raw = df.iloc[[row_idx]].copy()

    # Preprocess
    from datetime import datetime
    drop = ['Unnamed: 6','state_code.1','zip_code','id','Unnamed: 0','object_id','state_code']
    for c in drop:
        if c in raw.columns:
            raw.drop(columns=[c], inplace=True)

    raw['founded_at'] = pd.to_datetime(raw['founded_at'], errors='coerce')
    raw['closed_at']  = pd.to_datetime(raw['closed_at'], errors='coerce')

    if (raw['closed_at'] - raw['founded_at']).iloc[0] < pd.Timedelta(0):
        return jsonify({"ok": False, "error": "Invalid date order"}), 422

    ref = pd.to_datetime("2024-01-01")
    raw['closed_at'] = (ref - raw['closed_at']).dt.days.fillna(0).astype(int)

    if 'category_code' in raw.columns and CATEGORY_ENCODER_KEY in encoders:
        raw['category_code'] = encoders[CATEGORY_ENCODER_KEY].transform(raw['category_code'].astype(str))

    drop2 = ['status','founded_at','first_funding_at','last_funding_at','name','city']
    for c in drop2:
        if c in raw.columns:
            raw.drop(columns=[c], inplace=True)

    for col in feature_names:
        if col not in raw.columns:
            raw[col] = 0

    raw = raw[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)

    return jsonify({"ok": True, "values": raw.iloc[0].to_dict()})


# =========================
# Prediction Route
# =========================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    fields = feature_names[:]

    if request.method == "POST":
        row = {f: float(request.form.get(f, 0) or 0) for f in fields}
        X = pd.DataFrame([row], columns=fields)
        X_sc = scaler.transform(X)
        prob = float(model.predict(X_sc, verbose=0).ravel()[0])
        pred = "Closed / Failed" if prob >= 0.5 else "Acquired / Successful"
        return render_template("predict.html", fields=fields, prob=prob, pred=pred)

    return render_template("predict.html", fields=fields)



# =========================
# Run Local Dev
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

