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
