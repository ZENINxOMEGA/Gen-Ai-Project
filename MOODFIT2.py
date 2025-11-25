# ======================================================================================
# ||   MoodFit: Your Personal AI-Powered Outfit Recommender (Gemini API Version)       ||
# ======================================================================================

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
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "***********************************")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "*************************************")  # Replace if not using env variable

EMOTION_SAMPLING_SECONDS = 8
USE_MTCNN = True

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -------------------- LOCATION & WEATHER MODULE --------------------

def get_ip_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=6)
        data = response.json()

        if data["status"] != "success":
            raise RuntimeError("IP lookup failed")

        return float(data["lat"]), float(data["lon"]), data.get("city", "")
    
    except Exception as e:
        print(f"IP location lookup failed: {e}")
        return None, None, None



def fetch_weather(lat=None, lon=None, city=None):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"appid": OPENWEATHER_API_KEY, "units": "metric"}

    if lat is not None and lon is not None:
        params.update({"lat": lat, "lon": lon})
    elif city:
        params.update({"q": city})
    else:
        raise ValueError("You must provide either coordinates (lat,lon) or a city name.")

    response = requests.get(base_url, params=params, timeout=8)
    data = response.json()

    if data.get("cod") != 200:
        raise RuntimeError(f"Weather API returned an error: {data}")

    return {
        "city": data.get("name", "Unknown City"),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "condition": data["weather"][0]["main"],
        "description": data["weather"][0]["description"]
    }


# -------------------- EMOTION DETECTION MODULE --------------------

def detect_emotion(duration_sec=EMOTION_SAMPLING_SECONDS, display_cam=True):
    """
    Detects emotion using webcam and returns the dominant emotion.
    (This function does not print the "starting" banner — that is printed by caller
    so terminal ordering can be controlled.)
    """
    detector = FER(mtcnn=USE_MTCNN)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise RuntimeError("Could not access webcam. Check privacy settings.")


    emotion_accumulator = {k: 0.0 for k in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    last_detected_emotion = "neutral"
    start_time = time.time()

    while True:
        success, frame = cam.read()
        if not success:
            break

        detected_faces = detector.detect_emotions(frame)

        if detected_faces:
            main_face = detected_faces[0]
            emotions = main_face["emotions"]

            for e, val in emotions.items():
                emotion_accumulator[e] += val

            last_detected_emotion = max(emotions, key=emotions.get)

            if display_cam:
                x, y, w, h = main_face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {last_detected_emotion}",
                            (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if display_cam:
            cv2.imshow("MoodFit - Analyzing... (Press 'q' to stop)", frame)

        if (time.time() - start_time) >= duration_sec:
            break
        if display_cam and (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cam.release()
    if display_cam:
        cv2.destroyAllWindows()

    if sum(emotion_accumulator.values()) == 0:
        return last_detected_emotion

    return max(emotion_accumulator, key=emotion_accumulator.get)


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
        genai.configure(api_key="***********************************************")
        
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



# -------------------- VISUALIZATION --------------------

def show_fashion_samples(base_folder, categories, num_samples=6, thumb_size=(320,320), cols=3, caption=None):
    """
    Improved visualizer: show colorful images from local dataset with a polished grid UI.
    - base_folder: root dataset folder containing category subfolders (e.g., dataset/T-shirt_top/)
    - categories: list of category names (must match folder names or mapping)
    - num_samples: how many images PER category to display (total per category)
    - thumb_size: (w,h) size to resize thumbnail for display
    - cols: grid columns (images per row). If num_samples > cols, multiple rows produced.
    - caption: optional short text to show under the grid (e.g., the LLM recommendation line)
    """
    # mapping from FASHION_LABELS to local folder names (adjust if your names differ)
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

    # Collect images to display: (img_array, category_label, filename)
    grid_items = []
    for category in categories:
        folder_name = label_to_folder.get(category, category)
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            # skip missing categories silently
            continue

        image_files = [f for f in os.listdir(folder_path)if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]
        if not image_files:
            continue

        # choose up to num_samples images (randomize for variety)
        chosen = random.sample(image_files, min(num_samples, len(image_files)))
        for fn in chosen:
            p = os.path.join(folder_path, fn)
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # resize while keeping aspect ratio, pad to thumb_size
            h, w = img_rgb.shape[:2]
            target_w, target_h = thumb_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # create white background and center the resized image
            canvas = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
            x_off = (target_w - new_w)//2
            y_off = (target_h - new_h)//2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
            grid_items.append((canvas, category, fn))

    if not grid_items:
        print("[Info] No images available to display.")
        return

    # compute grid layout
    total = len(grid_items)
    rows = ceil(total / cols)
    fig_w = cols * (thumb_size[0] / 80)   # scale factor for figure size
    fig_h = rows * (thumb_size[1] / 120)
    plt.figure(figsize=(fig_w, fig_h + (0.4 if caption else 0.0)))
    plt.suptitle("Suggested Outfit Visuals", fontsize=18, fontweight='bold', y=0.98)

    for idx, (img, category, fn) in enumerate(grid_items):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        # category label below image (centered)
        ax.set_title(category, fontsize=10, fontweight='semibold')
        # add subtle border rectangle
        rect = Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#2b7a78", linewidth=2, clip_on=False)
        ax.add_patch(rect)

    # optional caption (original LLM text or explanation)
    if caption:
        plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=10, color="#333333")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, top=0.92)
    plt.show(block=True)

# -------------------- HELPER: PARSE OUTFIT TEXT --------------------

def extract_categories_from_text(outfit_text):
    """
    Return categories in LLM order, but resolve collisions:
      - If both T-shirt/top and Shirt detected, keep only T-shirt/top.
    """
    mapping_keywords = {
        "t-shirt":"T-shirt/top","tshirt":"T-shirt/top","tee":"T-shirt/top",
        "shirt":"Shirt",
        "trouser":"Trouser","trousers":"Trouser","pants":"Trouser","jeans":"Trouser",
        "pullover":"Pullover","sweater":"Pullover","jumper":"Pullover","hoodie":"Pullover",
        "dress":"Dress",
        "coat":"Coat","jacket":"Coat","blazer":"Coat",
        "sandal":"Sandal","flipflop":"Sandal",
        "sneaker":"Sneaker","trainers":"Sneaker",
        "boot":"Ankle boot","boots":"Ankle boot",
        "bag":"Bag","backpack":"Bag","purse":"Bag","tote":"Bag",
        "watch":"Watch"
    }

    # 1) check explicit VisualCategories line first (preserve order)
    m = re.search(r'VisualCategories\s*:\s*(.+)', outfit_text, flags=re.IGNORECASE)
    if m:
        listed = [c.strip() for c in m.group(1).split(',') if c.strip()]
        out = []
        for c in listed:
            if c in FASHION_LABELS and c not in out:
                out.append(c)
        if out:
            # resolve collision: prefer T-shirt/top over Shirt
            if "T-shirt/top" in out and "Shirt" in out:
                out = [x for x in out if x != "Shirt"]
            return out

    # 2) otherwise scan text left-to-right for keywords and labels
    s = outfit_text.lower()
    # normalize common variants to avoid double-matching later
    s = s.replace("t-shirt", "tshirt").replace("t-shirt", "tshirt")  # normalize hyphen variants

    found = []
    # detect explicit full-label occurrences (e.g., 't-shirt/top' may not appear verbatim; check label tokens)
    for label in FASHION_LABELS:
        lowlabel = label.lower()
        if lowlabel.replace("-", " ").replace("/", " ") in s and label not in found:
            found.append(label)

    # find keyword occurrences and record position
    positions = []
    for kw, lab in mapping_keywords.items():
        pos = s.find(kw)
        if pos != -1:
            positions.append((pos, lab))

    # combine and sort by first occurrence position
    # also include those already detected from full labels with pos=inf if no position found
    combined = []
    for i, lab in enumerate(found):
        combined.append((s.find(lab.lower().replace("-", " ").replace("/", " ")), lab))
    for pos, lab in positions:
        combined.append((pos, lab))

    # sort by position and collect unique labels preserving first-seen order
    combined_sorted = sorted(combined, key=lambda x: x[0] if x[0] >= 0 else float("inf"))
    ordered = []
    for pos, lab in combined_sorted:
        if lab not in ordered:
            ordered.append(lab)

    # Resolve collision: prefer T-shirt/top over Shirt
    if "T-shirt/top" in ordered and "Shirt" in ordered:
        ordered = [x for x in ordered if x != "Shirt"]

    # Last-resort fuzzy token scanning (keeps order)
    if not ordered:
        words = re.findall(r"[a-zA-Z\-]+", s)
        for w in words:
            match = difflib.get_close_matches(w, list(mapping_keywords.keys()), n=1, cutoff=0.85)
            if match:
                lab = mapping_keywords[match[0]]
                if lab not in ordered:
                    ordered.append(lab)

    return ordered



# -------------------- MAIN APP --------------------

def main():
    print("\n👋 Welcome to MoodFit (Gemini API Edition)!")
    print("================================================================")


    print("\nSTEP 1/3: Detecting your mood... 😃")

    sys.stdout.flush()
    time.sleep(0.15)

    try:
        emotion = detect_emotion()
        print(f"✅ Detected emotion: {emotion.capitalize()}")
    except Exception as e:
        print(f"❌ Error detecting emotion: {e}")
        sys.exit(1)

    print("\nSTEP 2/3: Fetching your local weather... 🌦")
    try:
        lat, lon, city = get_ip_location()
        weather = fetch_weather(lat=lat, lon=lon) if lat else fetch_weather(city=input("Enter city: "))
        print(f"✅ Weather in {weather['city']}: {weather['temp']}°C, {weather['description']}")
    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        sys.exit(1)

    print("\nSTEP 3/3: Generating outfit ideas... 👕")
    try:
        outfit_text, palette, reasons = get_llm_recommendation(emotion, weather)
        print("\n✨ Gemini AI Recommendations ✨")
        print(outfit_text)

        # -------------------- Show Visual Examples --------------------

        categories_to_show = extract_categories_from_text(outfit_text)
        if categories_to_show:
            print("\n🎨 Visual examples of suggested items:")
            sys.stdout.flush()
            # base folder is the 'dataset' folder next to this script
            base_folder = r"C:\Users\rohit sharma\OneDrive\Desktop\PROJECTS\JOVAC\dataset" 
            show_fashion_samples(base_folder=base_folder,
                                          categories=categories_to_show,
                                          num_samples=3,
                                          thumb_size=(250,250),
                                          cols=3,
                                          caption="Source: Gemini AI — suggestions shown above")
        else:
            print("[Info] Could not identify visual categories from LLM output. Showing representative images instead.")
            # fallback behavior: show an image for a few canonical categories (implement as needed)
            base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
            # choose some defaults (only if they exist)
            fallback = ["T-shirt/top", "Trouser", "Sneaker"]
            show_fashion_samples(base_folder=base_folder, categories=fallback, num_samples=1, thumb_size=(250,250), cols=3)

    except Exception as e:
        print(f"\n⚠ Gemini failed: {e}")
        sys.exit(1)

    print("\n✅ Done! Have a stylish day! 👗")


if __name__ == "__main__":
    main()
