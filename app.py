<<<<<<< HEAD
import streamlit as st
import google.generativeai as genai
import cv2
from PIL import Image
import numpy as np
import requests
from fer import FER
import time
import os
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import re
import difflib

# =========================
# CONFIG
# =========================
OPENWEATHER_API_KEY = "*********************************"
GEMINI_API_KEY = "********************************"

DATASET_PATH = r"C:\Users\rohit sharma\OneDrive\Desktop\PROJECTS\JOVAC\dataset"

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", "Watch"
]

# =========================
# FUNCTIONS
# =========================

def get_ip_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data["status"] != "success":
            return None, None, None
        return data["lat"], data["lon"], data["city"]
    except:
        return None, None, None


def fetch_weather(lat=None, lon=None, city=None):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"appid": OPENWEATHER_API_KEY, "units": "metric"}

    if lat and lon:
        params.update({"lat": lat, "lon": lon})
    elif city:
        params.update({"q": city})
    else:
        raise ValueError("Provide city or coordinates")

    r = requests.get(url, params=params)
    data = r.json()

    return {
        "city": data["name"],
        "temp": data["main"]["temp"],
        "description": data["weather"][0]["description"]
    }


def detect_emotion_from_image(uploaded_file):
    if uploaded_file is None:
        return None

    # Convert UploadedFile → PIL Image
    image = Image.open(uploaded_file)

    # Convert PIL → RGB numpy array
    img = np.array(image.convert("RGB"))

    detector = FER()
    emotion, score = detector.top_emotion(img)

    return emotion


def get_llm_recommendation(emotion, weather):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a professional fashion stylist in India.
Suggest a stylish and practical outfit based on:
Emotion: {emotion}
Weather: {weather['description']} at {weather['temp']}°C

Format:
👕 Top:
👖 Bottom:
👟 Footwear:
🧢 Accessories:
💬 Why it fits:
"""

    response = model.generate_content(prompt)
    return response.text


def extract_categories_from_text(outfit_text):
    s = outfit_text.lower()

    mapping = {
        # Tops
        "t-shirt": "T-shirt/top",
        "tee": "T-shirt/top",
        "top": "T-shirt/top",
        "kurti": "T-shirt/top",
        "cotton top": "T-shirt/top",
        "shirt": "Shirt",

        # Bottoms
        "pant": "Trouser",
        "pants": "Trouser",
        "jean": "Trouser",
        "jeans": "Trouser",
        "trouser": "Trouser",
        "palazzo": "Trouser",
        "palazzos": "Trouser",
        "wide-leg": "Trouser",

        # Winter wear
        "hoodie": "Pullover",
        "sweater": "Pullover",
        "pullover": "Pullover",
        "jacket": "Coat",
        "coat": "Coat",

        # Footwear
        "shoe": "Sneaker",
        "sneaker": "Sneaker",
        "sandals": "Sandal",
        "sandal": "Sandal",
        "slipper": "Sandal",
        "slippers": "Sandal",
        "juttis": "Sandal",
        "jutti": "Sandal",
        "flats": "Sandal",
        "flat sandals": "Sandal",

        # Accessories
        "bag": "Bag",
        "sling bag": "Bag",
        "boot": "Ankle boot",
        "boots": "Ankle boot",
        "watch": "Watch"
    }

    found = []

    for key, label in mapping.items():
        if key in s and label not in found:
            found.append(label)

    return found



def show_fashion_samples(categories, num_samples=3):
    label_to_folder = {
        "T-shirt/top": "T-shirt_top",
        "Trouser": "Trouser",
        "Pullover": "Pullover",
        "Dress": "Dress",
        "Coat": "Coat",
        "Sandal": "Sandal",
        "Shirt": "Shirt",
        "Sneaker": "Sneaker",
        "Bag": "Bag",
        "Ankle boot": "Ankle_boot",
        "Watch": "Watch"
    }

    images = []

    for cat in categories:
        folder = os.path.join(DATASET_PATH, label_to_folder.get(cat, ""))

        if not os.path.isdir(folder):
            st.warning(f"Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif"))]

        if not files:
            st.warning(f"No images found in {folder}")
            continue

        # Pick random samples
        selected = random.sample(files, min(num_samples, len(files)))

        for f in selected:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)

            # If image fails to load → skip
            if img is None:
                st.warning(f"Unreadable image skipped: {img_path}")
                continue

            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                st.warning(f"Could not convert image: {img_path}")
                continue

            images.append((img, cat))

    # ==========================
    # ⭐ DISPLAY CLEAN GRID
    # ==========================
    st.subheader("Visual Outfit Samples")

    for img, cat in images:
        col = st.columns(3)  # create NEW row every 3 images
        idx = 0
        break  # exit loop to move on

    # Now properly print all images
    for i, (img, cat) in enumerate(images):
        cols = st.columns(3)
        with cols[i % 3]:
            st.image(img, caption=cat, use_column_width=True)


# =========================
# UI STARTS HERE
# =========================

st.title("👗 MoodFit – AI Outfit Recommender")
st.subheader("Your emotion + local weather → Perfect outfit")

st.divider()

# 1️⃣ Capture Webcam Image
st.header("Step 1: Capture Your Mood")

image = st.camera_input("Click a selfie to analyse your emotion 😊")

if image:
    st.success("Image captured!")

    emotion = detect_emotion_from_image(image)
    st.info(f"Detected Emotion: **{emotion.upper()}**")

    # 2️⃣ Fetch Weather
    st.header("Step 2: Weather Info")

    lat, lon, city = get_ip_location()
    if lat:
        weather = fetch_weather(lat=lat, lon=lon)
    else:
        city = st.text_input("Enter your city:")
        if st.button("Fetch Weather"):
            weather = fetch_weather(city=city)

    if 'weather' in locals():
        st.success(f"Weather in {weather['city']}: {weather['temp']}°C, {weather['description']}")

        # 3️⃣ Generate Outfit
        st.header("Step 3: AI Outfit Recommendation")

        if st.button("Generate Outfit"):
            with st.spinner("Generating using Gemini..."):
                outfit = get_llm_recommendation(emotion, weather)

            st.success("Outfit generated!")
            st.write(outfit)

            # 4️⃣ Extract categories & show visual samples
            st.header("Visual Examples")
            categories = extract_categories_from_text(outfit)

            if categories:
                show_fashion_samples(categories)
            else:
                st.warning("Could not detect clothing items from AI response.")
=======

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Emotion model config ===
MODEL_FILENAME = "moodfit_emotion_model.h5"  # rename to your actual model file
MODEL_PATH = os.path.join(BASE_DIR, "model", MODEL_FILENAME)

# Adjust labels to match your own training
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

emotion_model = None

def load_emotion_model():
    global emotion_model
    if emotion_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Emotion model not found at {MODEL_PATH}. "
                "Place your trained model file there and update MODEL_FILENAME if needed."
            )
        emotion_model = load_model(MODEL_PATH)
    return emotion_model

# === Weather config ===
OPENWEATHER_API_KEY = "7d9dac083f86faab1c6d1f0ea76a1b4a"  # <-- replace with your key

def get_weather_for_city(city: str):
    """Call OpenWeatherMap API and return a compact dict."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "7d9dac083f86faab1c6d1f0ea76a1b4a":
        # For demo / development you can hardcode something instead of raising
        return {
            "city": city,
            "temp": 26.0,
            "description": "clear sky (demo)",
            "icon": "01d",
        }

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return {
        "city": city,
        "temp": float(data["main"]["temp"]),
        "description": data["weather"][0]["description"],
        "icon": data["weather"][0]["icon"],
    }

# === Outfit recommendation engine (intelligent rules) ===

def categorize_temp(temp_c: float) -> str:
    if temp_c <= 15:
        return "cold"
    if temp_c <= 25:
        return "mild"
    return "hot"

def get_outfit_recommendations(emotion: str, weather: dict):
    """Return a list of outfit suggestions based on emotion + weather."""
    temp = weather.get("temp", 24.0)
    temp_bucket = categorize_temp(temp)
    desc = weather.get("description", "").lower()

    suggestions = []

    def add_suggestion(title, description, items):
        suggestions.append(
            {
                "title": title,
                "description": description,
                "items": items,
                "emotion": emotion,
                "weather": weather,
            }
        )

    is_rainy = "rain" in desc or "drizzle" in desc or "storm" in desc
    is_sunny = "clear" in desc or "sun" in desc
    is_cloudy = "cloud" in desc

    em = emotion.lower()

    if em in ("happy", "excited", "surprise"):
        if temp_bucket == "hot":
            add_suggestion(
                "Vibrant Streetwear",
                "You seem full of energy! Go for light, breathable fabrics with bright accent colors that match your upbeat mood.",
                [
                    "Oversized graphic t‑shirt",
                    "Relaxed fit shorts or cargo joggers",
                    "Chunky sneakers or sporty trainers",
                    "Cap + minimal jewelry",
                ],
            )
        elif temp_bucket == "mild":
            add_suggestion(
                "Smart-Casual Glow",
                "Your happy mood + comfortable weather is perfect for a relaxed yet polished look.",
                [
                    "Well‑fitted jeans or chinos",
                    "Lightweight shirt or polo",
                    "White sneakers or loafers",
                    "Layer with a casual jacket or shrug",
                ],
            )
        else:
            add_suggestion(
                "Cozy Pop Layers",
                "Stay warm but keep your joyful vibe with a pop of color in your outerwear or accessories.",
                [
                    "Warm sweatshirt or knitted sweater",
                    "Dark jeans or wool trousers",
                    "Coat / puffer jacket in a fun color",
                    "Beanie + scarf",
                ],
            )

    elif em in ("sad", "down", "tired"):
        if temp_bucket == "hot":
            add_suggestion(
                "Easy-Breath Comfort",
                "Soft, loose outfits to keep your body relaxed while your mind recharges.",
                [
                    "Loose cotton t‑shirt",
                    "Soft joggers or relaxed shorts",
                    "Slip‑on shoes or sliders",
                    "Light neutral colors (beige, pastel blue, sage)",
                ],
            )
        else:
            add_suggestion(
                "Soft Comfort Layers",
                "Comfort‑first outfits with gentle fabrics to help you feel safe and grounded.",
                [
                    "Oversized hoodie or cardigan",
                    "Joggers / leggings / relaxed fit jeans",
                    "Warm socks + sneakers",
                    "Muted, calming tones (greys, navy, forest green)",
                ],
            )

    elif em in ("angry", "frustrated", "annoyed"):
        add_suggestion(
            "Clean Minimal Power Look",
            "Structured, minimal outfits can help you feel more in control and less chaotic.",
            [
                "Solid color t‑shirt or shirt (black/white/charcoal)",
                "Straight‑fit jeans or trousers",
                "Simple sneakers or boots",
                "Minimal accessories, clean lines",
            ],
        )

    elif em in ("fear", "anxious", "worried"):
        add_suggestion(
            "Grounding Essentials",
            "Soft layers and secure footwear to help your body feel safe and supported.",
            [
                "Soft knit top or sweatshirt",
                "Comfortable jeans / joggers",
                "Closed shoes with good grip",
                "Optional: a familiar jacket you love",
            ],
        )

    else:
        # Neutral / unknown
        add_suggestion(
            "Balanced Everyday Fit",
            "A versatile look that works in most situations while staying comfortable.",
            [
                "Plain t‑shirt or casual shirt",
                "Jeans / chinos",
                "Sneakers",
                "Optional light jacket depending on weather",
            ],
        )

    # Weather-specific overlays
    if is_rainy:
        add_suggestion(
            "Rain‑Ready Layered Fit",
            "Since it might rain, stay dry without losing style.",
            [
                "Water‑resistant jacket or hoodie",
                "Quick‑dry pants or dark jeans",
                "Water‑proof sneakers / boots",
                "Umbrella or cap",
            ],
        )
    elif is_sunny and temp_bucket == "hot":
        add_suggestion(
            "Sun‑Shield Summer Fit",
            "Protect yourself from the sun while staying fresh and stylish.",
            [
                "Light linen or cotton shirt",
                "Shorts or loose trousers",
                "Sunglasses + cap",
                "Breathable sneakers or sandals",
            ],
        )
    elif is_cloudy and temp_bucket != "hot":
        add_suggestion(
            "Cloudy Day Casual",
            "Soft layers for slightly cool, cloudy weather.",
            [
                "Long‑sleeve tee or henley",
                "Jeans / joggers",
                "Sneakers",
                "Light jacket / overshirt",
            ],
        )

    if not suggestions:
        add_suggestion(
            "Fallback Everyday Look",
            "We could not match a specific rule, so here is a safe everyday outfit.",
            [
                "Basic tee",
                "Neutral jeans",
                "Sneakers",
            ],
        )

    return suggestions

# === Image preprocessing & prediction ===

def preprocess_image(image_path: str):
    """Preprocess image for VGG19-style model (224x224 RGB). Adjust if your model differs."""
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_emotion(image_path: str) -> str:
    model = load_emotion_model()
    x = preprocess_image(image_path)
    preds = model.predict(x)
    if preds.ndim == 2:
        idx = int(np.argmax(preds[0]))
    else:
        idx = int(np.argmax(preds))
    if 0 <= idx < len(EMOTION_LABELS):
        return EMOTION_LABELS[idx]
    return "Neutral"

# === Routes ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    # 1. Get image
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2. Save to a temporary path
    uploads_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    filename = secure_filename(img_file.filename)
    img_path = os.path.join(uploads_dir, filename)
    img_file.save(img_path)

    # 3. Get city & weather
    city = request.form.get("city", "Delhi")
    try:
        weather = get_weather_for_city(city)
    except Exception as e:
        weather = {
            "city": city,
            "temp": 24.0,
            "description": f"weather-unavailable ({e})",
            "icon": "",
        }

    # 4. Predict emotion
    try:
        emotion = predict_emotion(img_path)
    except Exception as e:
        return jsonify({"error": f"Emotion prediction failed: {e}"}), 500
    finally:
        # Best effort cleanup
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass

    # 5. Get recommendations
    outfits = get_outfit_recommendations(emotion, weather)

    return jsonify(
        {
            "emotion": emotion,
            "weather": weather,
            "outfits": outfits,
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> ce239939 (just added backend to the frontend)
