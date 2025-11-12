# ======================================================================================
# ||   MoodFit: Your Personal AI-Powered Outfit Recommender                        ||
# ======================================================================================
#
#  Ever stare at your closet and think, "I have nothing to wear"?
#  MoodFit is a fun project that tries to solve that problem by suggesting what to wear
#  based on two things that really influence our choices: your current mood and the
#  local weather.
#
#  How it works:
#  1.  It activates your webcam to see your facial expression and guess your emotion.
#  2.  It finds your location to get the current weather conditions.
#  3.  A simple "Rule Engine" combines your mood and the weather to recommend
#      clothing categories from the popular Fashion-MNIST dataset.
#
#  Author: OMEGA (GLA University) -- A Gen-AI Project
# --------------------------------------------------------------------------------------

import os
import time
import sys
import requests
import numpy as np
import cv2
from fer import FER
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present 

# -------------------- CONFIGURATION --------------------
# You can tweak these settings to change the app's behavior.

OPENWEATHER_API_KEY = os.getenv("open_ai")  # Get your API key from https://openweathermap.org/api

# How long (in seconds) the app should analyze your emotion from the webcam.
EMOTION_SAMPLING_SECONDS = 8

# The FER library can use a more advanced face detector called MTCNN.
# It's more accurate but might be slower. Set to False to use the default one.
USE_MTCNN = True

# These are the 10 clothing categories our recommender knows about,
# based on the Fashion-MNIST dataset.
FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -------------------- LOCATION & WEATHER MODULE --------------------

def get_ip_location(timeout=5):
    """
    Try to fetch location from IP using multiple fallback services.
    """
    
    # --- Service 1: ipapi.co (Primary) ---
    try:
        url_1 = "https://ipapi.co/json/"
        print("  > Attempting location lookup with ipapi.co...")
        response = requests.get(url_1, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()
        
        lat = data.get("latitude")
        lon = data.get("longitude")
        city = data.get("city")
        
        if lat and lon:
            print(f"  > Success! Location found via ipapi.co: {city}")
            return float(lat), float(lon), city or ""
        
    except Exception as e:
        print(f"  > Info: Service 1 (ipapi.co) failed: {e}. Trying fallback.")

    # --- Service 2: ip-api.com (Fallback) ---
    try:
        url_2 = "http://ip-api.com/json/" # Note: HTTP for this free endpoint
        print("  > Attempting location lookup with ip-api.com...")
        response = requests.get(url_2, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            lat = data.get("lat")
            lon = data.get("lon")
            city = data.get("city")
            
            if lat and lon:
                print(f"  > Success! Location found via ip-api.com: {city}")
                return float(lat), float(lon), city or ""
                
    except Exception as e:
        print(f"  > Info: Service 2 (ip-api.com) also failed: {e}.")

    # ✅ If all services fail
    print("IP location lookup failed after all attempts.")
    return None, None, None


def fetch_weather(lat=None, lon=None, city=None):
    """
    Gets the current weather from the OpenWeatherMap API for a given location.
    Returns a dictionary with key weather details like temperature and conditions.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"appid": OPENWEATHER_API_KEY, "units": "metric"} # Use Celsius

    if lat is not None and lon is not None:
        params.update({"lat": lat, "lon": lon})
    elif city:
        params.update({"q": city})
    else:
        # We can't get weather without a location!
        raise ValueError("You must provide either coordinates (lat,lon) or a city name.")

    response = requests.get(base_url, params=params, timeout=8)
    data = response.json()

    # The API returns a 'cod' (code) field. 200 means "OK". Anything else is an error.
    if data.get("cod") != 200:
        raise RuntimeError(f"Weather API returned an error: {data}")

    # We'll pick out just the useful bits of information to return.
    return {
        "city": data.get("name", "Unknown City"),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "condition": data["weather"][0]["main"],       # e.g., "Clear", "Rain", "Clouds"
        "description": data["weather"][0]["description"] # e.g., "light rain"
    }

# -------------------- EMOTION DETECTION MODULE --------------------

def detect_emotion(duration_sec=EMOTION_SAMPLING_SECONDS, display_cam=True):
    """
    Turns on the webcam, detects faces, and analyzes facial expressions.
    It aggregates emotion scores over a few seconds to find the most dominant emotion.
    """
    print("Starting webcam... Look at the camera and express yourself!")
    detector = FER(mtcnn=USE_MTCNN)
    cam = cv2.VideoCapture(0) # 0 is usually the default built-in webcam.

    if not cam.isOpened():
        raise RuntimeError("Could not access your webcam. Is it being used by another app? Or check your privacy settings.")

    # A dictionary to add up the scores for each emotion over time.
    emotion_accumulator = {k: 0.0 for k in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    last_detected_emotion = "neutral"
    start_time = time.time()

    while True:
        is_frame_read_successfully, frame = cam.read()
        if not is_frame_read_successfully:
            break

        # The 'detect_emotions' function returns a list of faces found in the frame.
        detected_faces = detector.detect_emotions(frame)

        if detected_faces:
            # We'll focus on the most prominent face detected.
            main_face_result = detected_faces[0]
            emotion_scores = main_face_result["emotions"]

            # Add the scores from this frame to our running total.
            for emotion, score in emotion_scores.items():
                if emotion in emotion_accumulator:
                    emotion_accumulator[emotion] += float(score)

            # Find the strongest emotion in this single frame to display it.
            last_detected_emotion = max(emotion_scores, key=emotion_scores.get)

            if display_cam:
                # Draw a green box around the detected face.
                x, y, w, h = main_face_result['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display the current dominant emotion.
                cv2.putText(frame, f"Emotion: {last_detected_emotion}",
                            (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if display_cam:
            cv2.imshow("MoodFit - Analyzing... (Press 'q' to finish early)", frame)

        # Check if we've run long enough or if the user wants to quit.
        if (time.time() - start_time) >= duration_sec:
            break
        if display_cam and (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # Clean up by releasing the webcam and closing the window.
    cam.release()
    if display_cam:
        cv2.destroyAllWindows()

    # If we never detected any emotions, just return the default.
    if sum(emotion_accumulator.values()) == 0:
        return last_detected_emotion

    # Find the emotion with the highest total score over the entire duration.
    dominant_emotion = max(emotion_accumulator, key=emotion_accumulator.get)
    return dominant_emotion

# -------------------- RECOMMENDATION ENGINE --------------------

def recommend_outfits(emotion, weather_data):
    """
    The core logic of the app. It takes the detected emotion and weather,
    and uses a set of simple rules to recommend outfit categories.
    """
    scores = {category: 0.0 for category in FASHION_LABELS}
    temperature = weather_data["temp"]
    weather_condition = weather_data["condition"]

    # --- Part 1: Weather-Driven Rules ---
    # These rules adjust scores based on temperature and conditions.

    # Based on temperature
    if temperature < 10:  # Cold
        for item in ["Coat", "Pullover", "Trouser", "Ankle boot", "Sneaker"]: scores[item] += 2.0
    elif temperature < 20: # Cool
        for item in ["Coat", "Pullover", "Trouser", "Sneaker", "Shirt"]: scores[item] += 1.5
    elif temperature < 28: # Warm
        for item in ["T-shirt/top", "Shirt", "Dress", "Sneaker", "Trouser"]: scores[item] += 1.5
    else:  # Hot
        for item in ["T-shirt/top", "Sandal", "Dress", "Trouser"]: scores[item] += 2.0

    # Based on weather condition
    if weather_condition in {"Rain", "Thunderstorm", "Drizzle"}:
        for item in ["Coat", "Trouser", "Sneaker", "Ankle boot", "Bag"]: scores[item] += 1.5
    elif weather_condition == "Clear":
        for item in ["Sandal", "T-shirt/top", "Dress", "Shirt"]: scores[item] += 1.0
    elif weather_condition == "Snow":
        for item in ["Coat", "Pullover", "Ankle boot", "Trouser"]: scores[item] += 2.0

    # --- Part 2: Emotion-Driven Rules ---
    # These rules add points based on the user's mood.

    if emotion in {"happy", "surprise"}:
        # Suggest more expressive or casual clothing for positive moods.
        for item in ["Dress", "T-shirt/top", "Sandal", "Shirt", "Sneaker"]: scores[item] += 1.5
        suggested_colors = ["vibrant colors", "pastels", "warm tones"]
    elif emotion in {"sad", "fear"}:
        # Suggest comfortable, cozy items for sad or anxious moods.
        for item in ["Pullover", "Coat", "Trouser", "Sneaker"]: scores[item] += 1.5
        suggested_colors = ["comforting neutrals", "earth tones", "calming blues"]
    elif emotion in {"angry", "disgust"}:
        # Suggest simple, non-restrictive clothing.
        for item in ["Trouser", "Pullover", "Sneaker", "Coat"]: scores[item] += 1.2
        suggested_colors = ["cool blues", "greens", "shades of grey"]
    else:  # neutral
        # Suggest versatile, classic items.
        for item in ["Shirt", "Trouser", "Sneaker", "T-shirt/top"]: scores[item] += 1.0
        suggested_colors = ["classic neutrals", "monochrome palettes"]

    # --- Part 3: Finalizing the Recommendation ---
    # Sort the items by their final scores to find the top 3.
    top_3_recommendations = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]

    # Generate a simple, user-friendly reason for each recommendation.
    reasons_list = []
    for category, score in top_3_recommendations:
        reason = "A versatile and practical choice for today." # Default reason
        if category in {"Coat", "Pullover", "Ankle boot"} and temperature < 20:
            reason = "A great choice to stay warm and cozy."
        elif category == "Sandal" and temperature >= 24 and weather_condition == "Clear":
            reason = "Perfect for keeping cool on a warm, clear day."
        elif category in {"Dress", "T-shirt/top", "Shirt"} and temperature >= 18:
            reason = "Light and comfortable for the mild weather."
        reasons_list.append({"category": category, "reason": reason})

    return top_3_recommendations, suggested_colors, reasons_list

# -------------------- BONUS VISUALIZATION --------------------

def show_fashion_samples(categories, num_samples=6):
    """
    A fun bonus function that loads the Fashion-MNIST dataset and
    displays a few example images for the recommended clothing categories.
    """
    try:
        (x_train, y_train), _ = fashion_mnist.load_data()
    except Exception as e:
        print(f"\n[Info] Could not load Fashion-MNIST sample images: {e}")
        return

    label_to_index = {name: i for i, name in enumerate(FASHION_LABELS)}

    for category in categories:
        index = label_to_index.get(category)
        if index is None: continue

        # Find the first few images in the dataset that match the recommended category.
        indices = np.where(y_train == index)[0][:num_samples]
        plt.figure(figsize=(8, 2))

        for i, image_index in enumerate(indices):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x_train[image_index], cmap="gray")
            plt.axis("off") # Hide the x/y axis labels.

        plt.suptitle(f"Examples of: {category}", fontsize=14)
        plt.show()


# -------------------- MAIN APPLICATION FLOW --------------------
def main():
    """The main function that runs the entire MoodFit application sequence."""
    print("\n👋 Welcome to MoodFit! Let's find the perfect outfit for you.")
    print("================================================================")

    # --- Step 1: Detect Emotion ---
    print("\nSTEP 1/3: Analyzing your mood... 😃")
    try:
        dominant_emotion = detect_emotion()
        print(f"✅ Your dominant emotion appears to be: {dominant_emotion.capitalize()}")
    except Exception as e:
        print(f"❌ Oh no! Could not detect emotion. Error: {e}")
        sys.exit(1)

    # --- Step 2: Get Weather ---
    print("\nSTEP 2/3: Checking your local weather... 🌦️")
    try:
        lat, lon, city = get_ip_location()
        if lat is not None:
            weather_data = fetch_weather(lat=lat, lon=lon)
        else:
            print("Could not automatically determine your location.")
            manual_city = input("Please enter your city name: ").strip()
            if not manual_city:
                print("No city provided. Exiting.")
                sys.exit(1)
            weather_data = fetch_weather(city=manual_city)
        
        print(f"✅ Weather for {weather_data['city']}: {weather_data['temp']}°C (feels like {weather_data['feels_like']}°C) with {weather_data['description']}.")

    except Exception as e:
        print(f"❌ Oops! Could not fetch the weather. Error: {e}")
        print("   Please check your internet connection and that the API key is correct.")
        sys.exit(1)

    # --- Step 3: Get Recommendations ---
    print("\nSTEP 3/3: Generating your personalized outfit ideas... 👕")
    top_items, palette, reasons = recommend_outfits(dominant_emotion, weather_data)

    print("\n✨ Here are your top 3 recommendations for today: ✨")
    for r in reasons:
        print(f"  • {r['category']:<12} — {r['reason']}")
    print(f"\n🎨 Suggested color palette to match your mood: {', '.join(palette)}.")

    # --- Step 4 (Optional): Show Image Samples ---
    show_samples = input("\nWould you like to see some visual examples? (y/n): ").strip().lower()
    if show_samples == 'y':
        try:
            recommended_categories = [item for item, score in top_items]
            show_fashion_samples(recommended_categories)
        except Exception as e:
            print(f"Sorry, could not display sample images: {e}")

    print("\n================================================================")
    print("✅ All done! Hope you have a great day. Close any image windows to exit.")
    print("================================================================\n")

# This standard Python construct ensures that the `main()` function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()