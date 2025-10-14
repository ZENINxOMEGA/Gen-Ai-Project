// Simulate Mood Detection + Weather Fetch + Outfit Suggestion 💡

document.getElementById('processBtn').addEventListener('click', () => {
  const moods = ['Happy', 'Sad', 'Confident', 'Excited', 'Chill'];
  const mood = moods[Math.floor(Math.random() * moods.length)];
  document.getElementById('detectedMood').textContent = `Detected Mood: ${mood}`;
  generateOutfit(mood, window.currentWeatherType || 'Sunny');
});

// Auto-fetch weather
window.addEventListener('load', () => {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(async pos => {
      const { latitude, longitude } = pos.coords;
      const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`);
      const data = await res.json();
      const temp = data.current_weather.temperature;
      const weather = data.current_weather.weathercode < 3 ? "Sunny" : "Cloudy";
      window.currentWeatherType = weather;
      document.getElementById('weatherInfo').textContent = `${weather}, ${temp}°C`;
    });
  }
});

// Outfit logic
function generateOutfit(mood, weather) {
  const outfitText = document.getElementById('outfitText');
  const outfitImg = document.getElementById('outfitImg');
  let suggestion = "";
  let img = "assets/outfits/casual.jpg";

  if (weather === "Rainy") {
    suggestion = `It's ${weather} and you're feeling ${mood} — try a waterproof jacket with bright sneakers!`;
    img = "assets/outfits/sporty.jpg";
  } else if (mood === "Confident") {
    suggestion = `You're ${mood} today — go for a sharp blazer and sleek boots.`;
    img = "assets/outfits/formal.jpg";
  } else {
    suggestion = `Since it's ${weather} and you're ${mood}, try a comfy casual look!`;
  }

  outfitText.textContent = suggestion;
  outfitImg.src = img;
}

// Modal toggle
function showModal(id) {
  document.getElementById(id).style.display = 'flex';
}

function hideModals() {
  document.querySelectorAll('.modal').forEach(m => m.style.display = 'none');
}

document.getElementById('loginBtn').onclick = () => showModal('loginModal');
document.getElementById('signupBtn').onclick = () => showModal('signupModal');
document.querySelectorAll('.modal').forEach(m => m.addEventListener('click', e => {
  if (e.target === m) hideModals();
}));


// Scroll reveal animations
document.querySelectorAll('nav a').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const href = this.getAttribute('href');
    if (href.startsWith('#')) {
      e.preventDefault();
      document.querySelector(href).scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});


ScrollReveal({
  distance: '60px',
  duration: 1500,
  delay: 100,
  easing: 'ease-in-out',
  reset: false 
});

ScrollReveal().reveal('.header', { origin: 'top' });
ScrollReveal().reveal('.about h2, .about p, .infographic div', { origin: 'bottom', interval: 100 });
ScrollReveal().reveal('.core h2', { origin: 'top' });
ScrollReveal().reveal('.upload-area, .weather-card, .suggestion-card', { origin: 'bottom', interval: 150 });
ScrollReveal().reveal('.modal-content', { origin: 'top', delay: 200 });
ScrollReveal().reveal('footer', { origin: 'bottom', delay: 150 });


/* ---------------------------
   LOGIN/SIGNUP SLIDE EFFECT
---------------------------- */
const slideBg = document.querySelector('.auth-slide .slide-bg');
const loginBtn = document.getElementById('loginBtn');
const signupBtn = document.getElementById('signupBtn');

// Default: highlight "Login"
slideBg.style.left = "0%";

loginBtn.addEventListener('mouseenter', () => {
  slideBg.style.left = "0%";
});
signupBtn.addEventListener('mouseenter', () => {
  slideBg.style.left = "50%";
});

// Reset on mouse leave (optional)
document.querySelector('.auth-slide').addEventListener('mouseleave', () => {
  slideBg.style.left = "0%";
});