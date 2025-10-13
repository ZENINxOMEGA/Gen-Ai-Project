# 👗 MoodFit – Virtual Stylist (Outfit Recommendation Based on Mood & Weather)

> **Your Personal AI Stylist!**  
> MoodFit suggests the perfect outfit based on your *mood* and *current weather*, blending fashion with intelligence.  

---

## 🧠 About The Project
**MoodFit** is a GenAI-based virtual stylist that understands how you *feel* and what the *weather* is like — then recommends the best outfit to match your vibe and comfort.  
Whether you're feeling happy, calm, energetic, or moody — MoodFit ensures you stay confident *and* stylish.

---

## ✨ Key Features
- 🌤️ Real-time **Weather Detection** using API  
- 😊 **Mood-based Recommendations** via user input or expression  
- 🧥 Intelligent **Rules-based Recommendation Engine** (upgradable to LLM-based reasoning)  
- 💡 Responsive **Frontend Interface** (HTML, CSS, JS)  
- 🧩 Modular Python backend for easy feature integration  
- 🎨 Clean UI with dynamic outfit suggestions  

---

## 🛠️ Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python |
| **APIs Used** | OpenWeather API (for weather data) |
| **Recommendation System** | Rule-Based Engine → *(Future upgrade: LLM-powered reasoning)* |
| **Version Control** | Git, GitHub |

---

## 🧭 Project Workflow
```mermaid
graph TD
A[User Input] --> B[Mood Selection or Detection]
B --> C[Fetch Weather via API]
C --> D[Rules-Based Recommendation Engine]
D --> E[Generate Outfit Suggestions]
E --> F[Display Stylish Recommendations on UI]
