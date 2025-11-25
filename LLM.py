import random
from math import ceil
from matplotlib.patches import Rectangle
import difflib
import re
import os
import time
import sys
import json
import requests
import numpy as np
import cv2
from fer import FER
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import google.generativeai as genai # <-- Gemini
import io

# ✅ Force UTF-8 for Windows consoles (fixes UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.system("chcp 65001 >NUL")

# -------------------- CONFIGURATION --------------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "c84e037d60c284b9877c3fb3d80d6b51")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBYSduLyKKMadzSIr7tvg9mNyDqGp89q6Y")  # Replace if not using env variable

EMOTION_SAMPLING_SECONDS = 8
USE_MTCNN = True

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -------------------- GEMINI LLM OUTFIT RECOMMENDER --------------------

def get_llm_recommendation(emotion, weather):
    """
    Generate outfit suggestions using Google's Gemini API via the Google GenAI SDK.
    Returns (text, palette_list, reasons_list)
    """

    # Load API key from environment if available
    api_key = GEMINI_API_KEY if GEMINI_API_KEY and "****" not in GEMINI_API_KEY else None
    
    try:
        # Configure Gemini SDK
        genai.configure(api_key="AIzaSyBYSduLyKKMadzSIr7tvg9mNyDqGp89q6Y")
        
        # Create Gemini model (NO CLIENT in new SDK)
        model = genai.GenerativeModel("gemini-2.5-flash")

    except Exception as e:
        raise RuntimeError(f"Could not initialize Gemini client: {e}")

    prompt = f"""
You are a professional fashion stylist in India.
Suggest a stylish and practical outfit based on the following:
- Emotion: {emotion}
- Weather: {weather['description']} at {weather['temp']}°C
- Temperature guideline for India:
    • Below 16°C → very cool:
    • 16–22°C → cool: 
    • 22–28°C → pleasant: 
    • Above 28°C → hot: 
suggestions should consider common clothing styles in India.
and local cultural preferences.
and be suitable for everyday wear.
and practical for the weather conditions.
and comfortable.
and easy to find in local Indian markets.

Format exactly like:
👕 Top:
👖 Bottom:
👟 Footwear:
🧢 Accessories:
💬 Why it fits:
Keep it short and practical.
IMPORTANT: For mapping to visuals, prefer items from this list when possible:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot, Watch.
"""

    try:
        # Generate content
        response = model.generate_content(prompt)

        # Extract text
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Empty response from Gemini model.")

        return text.strip(), [], [{"category": "Gemini Suggestion", "reason": "Generated via Gemini AI"}]

    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")


