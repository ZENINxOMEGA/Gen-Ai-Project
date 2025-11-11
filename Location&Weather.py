import os
import time
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt

OPENWEATHER_API_KEY = "********************************"  

# -------------------- LOCATION & WEATHER MODULE --------------------
sys.stdout.reconfigure(encoding='utf-8')
def get_ip_location():
    """
    Tries to find your location (latitude, longitude, city) based on your IP address.
    This is a quick and easy way to get location without asking the user.
    """
    try:
        # A simple API that returns location data for your IP address.
        response = requests.get("https://ipapi.co/json/", timeout=6)
        data = response.json()
        return float(data["latitude"]), float(data["longitude"]), data.get("city", "")
    except Exception as e:
        # If it fails (e.g., no internet, API is down), we'll just return nothing.
        print(f"IP location lookup failed: {e}")
        return None, None, None

def fetch_weather(lat=None, lon=None, city=None):
    """
    Gets the current weather from the OpenWeatherMap API for a given location.
    Returns a dictionary with key weather details like temperature and conditions.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    query_data = {"appid": OPENWEATHER_API_KEY, "units": "metric"} # Use Celsius

    if lat is not None and lon is not None:
        query_data.update({"lat": lat, "lon": lon})
    elif city:
        query_data.update({"q": city})
    else:
        # We can't get weather without a location!
        raise ValueError("You must provide either coordinates (lat,lon) or a city name.")

    response = requests.get(base_url, params=query_data, timeout=8)
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


