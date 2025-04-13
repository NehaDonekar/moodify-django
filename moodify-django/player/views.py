from django.shortcuts import render
import os, random, pickle
from django.conf import settings

# Load model and vectorizer
BASE_DIR = settings.BASE_DIR
model = pickle.load(open(os.path.join(BASE_DIR, 'mood_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'mood_vectorizer.pkl'), 'rb'))

def home(request):
    song_path = None
    predicted_mood = None

    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text:
            vectorized = vectorizer.transform([text])
            predicted_mood = model.predict(vectorized)[0].lower()

            request.session['predicted_mood'] = predicted_mood  # 🔁 Save to session

            songs_dir = os.path.join(BASE_DIR, 'static', 'songs', predicted_mood)
            if os.path.exists(songs_dir):
                songs = [s for s in os.listdir(songs_dir) if s.endswith('.mp3')]
                if songs:
                    selected_song = random.choice(songs)
                    song_path = os.path.join('songs', predicted_mood, selected_song)
                    request.session['song_path'] = song_path  # 🔁 Save to session
                else:
                    request.session['song_path'] = None
            else:
                request.session['song_path'] = None

    else:
        # 🔁 Load from session if exists
        predicted_mood = request.session.get('predicted_mood')
        song_path = request.session.get('song_path')

    return render(request, 'player/home.html', {
        'song_path': song_path,
        'predicted_mood': predicted_mood
    })
