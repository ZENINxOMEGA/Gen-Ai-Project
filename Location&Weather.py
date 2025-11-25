import os
import time
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt

OPENWEATHER_API_KEY = "*********************************"  

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





