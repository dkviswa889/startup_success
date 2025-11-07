import os
os.environ["KERAS_BACKEND"] = "numpy"   # ✅ Force Keras to run without TensorFlow

import json
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

# =========================
# App
# =========================
app = Flask(__name__)
app.secret_key = APP_SECRET
app.permanent_session_lifetime = timedelta(hours=6)

CSV_PATH = r"startup data.csv"

# =========================
# Load model + preproc
# =========================
try:
    model = load_model(MODEL_PATH)  # ✅ Uses Keras directly
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if not feature_names:
        raise AttributeError("scaler has no feature_names_in_.")
except Exception as e:
    feature_names = []
    print(f"[WARN] Could not infer feature names from scaler: {e}")

try:
    encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    encoders = {}
    print(f"[WARN] Could not load label encoders: {e}")

CATEGORY_ENCODER_KEY = "category_code"

# =========================
# Auth helpers
# =========================
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

# =========================
# Routes
# =========================
@app.route("/")
def root():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pw   = request.form.get("password", "").strip()
        if user == ADMIN_USER and pw == ADMIN_PASS:
            session["logged_in"] = True
            session["username"] = user
            flash("Welcome back!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/index")
@login_required
def index():
    return render_template("index.html")

def _preprocess_single_row(raw_row: pd.DataFrame) -> pd.DataFrame:
    row = raw_row.copy()

    for c in ['Unnamed: 6','state_code.1','zip_code','id','Unnamed: 0','object_id','state_code']:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    row['founded_at'] = pd.to_datetime(row['founded_at'], errors='coerce')
    row['closed_at']  = pd.to_datetime(row['closed_at'],  errors='coerce')
    row['first_funding_at'] = pd.to_datetime(row['first_funding_at'], errors='coerce')
    row['last_funding_at']  = pd.to_datetime(row['last_funding_at'],  errors='coerce')

    mask_bad = (row['closed_at'] - row['founded_at'] < pd.Timedelta(0))
    if mask_bad.iloc[0]:
        raise ValueError("closed_at occurs before founded_at.")

    ref_date = pd.to_datetime('2024-01-01')
    row['closed_at'] = (ref_date - row['closed_at']).dt.days.fillna(0).astype(int)

    for c in ['age_first_milestone_year','age_last_milestone_year']:
        if c in row.columns:
            row[c] = row[c].fillna(0)

    if 'category_code' in row.columns:
        if encoders and 'category_code' in encoders:
            row['category_code'] = encoders['category_code'].transform(row['category_code'].astype(str))
        else:
            raise ValueError("Missing label encoder for category_code")

    for c in ['status','founded_at','first_funding_at','last_funding_at','name','city']:
        if c in row.columns:
            row.drop(columns=[c], inplace=True)

    if not feature_names:
        raise ValueError("Missing feature names from scaler")

    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_names]
    row = row.apply(pd.to_numeric, errors='coerce').fillna(0)

    return row

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    int_like = set([
        "relationships","funding_rounds","funding_total_usd","milestones",
        "is_top500","founded_at_year","founded_at_day","founded_at_month",
        "first_funding_at_year","first_funding_at_day","first_funding_at_month",
        "last_funding_at_year","last_funding_at_day","last_funding_at_month",
        "closed_at", "has_VC","has_angel","has_roundA","has_roundB","has_roundC","has_roundD"
    ])

    fields = feature_names[:] if feature_names else []
    has_category_text = CATEGORY_ENCODER_KEY in fields and CATEGORY_ENCODER_KEY in encoders

    if request.method == "POST":
        row = {}
        errors = []

        if has_category_text:
            cat_text = request.form.get("category_code_text", "").strip()
            if cat_text:
                try:
                    le = encoders[CATEGORY_ENCODER_KEY]
                    row[CATEGORY_ENCODER_KEY] = int(le.transform([cat_text])[0])
                except Exception:
                    errors.append(f"Unknown category label: '{cat_text}'.")

        for f in fields:
            if f == CATEGORY_ENCODER_KEY and has_category_text and f in row:
                continue
            val = request.form.get(f, "").strip()
            if val == "":
                row[f] = 0
                continue
            try:
                row[f] = int(float(val)) if f in int_like else float(val)
            except ValueError:
                errors.append(f"Field '{f}' must be numeric.")

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("predict.html", fields=fields, has_category_text=has_category_text)

        X = pd.DataFrame([row], columns=fields)

        try:
            X_sc = scaler.transform(X)
            prob = float(model.predict(X_sc, verbose=0).ravel()[0])
            pred = int(prob >= 0.5)
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            return render_template("predict.html", fields=fields, has_category_text=has_category_text)

        label_map = {0: "Acquired/Successful", 1: "Closed/Not Acquired"}
        flash(f"Predicted probability: {prob:.3f} → {label_map.get(pred, pred)}", "success")
        return render_template("predict.html", fields=fields, has_category_text=has_category_text, prob=prob, pred=pred)

    return render_template("predict.html", fields=fields, has_category_text=has_category_text)

@app.route("/_features")
@login_required
def _features():
    return jsonify({"feature_names": feature_names})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

