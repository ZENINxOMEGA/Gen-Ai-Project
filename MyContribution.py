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
    print("\nSTEP 2/3: Checking your local weather... 🌦")
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
        print(f" Oops! Could not fetch the weather. Error: {e}")
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

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()