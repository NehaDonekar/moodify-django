import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# 💡 Expanded training set with realistic inputs
data = {
    'text': [
        # Happy
        "I am very happy", "joyful and excited", "feeling awesome", "life is amazing", "I am thrilled",
        # Sad
        "so sad and alone", "feeling depressed", "I miss you", "tears in my eyes", "I am so down and emotional",
        # Devotional
        "spiritual mood", "connected to God", "peaceful prayer", "divine energy", "meditating to God",
        # Nervous
        "I am nervous", "feeling anxious", "shaking hands", "I'm scared", "I'm tensed"
    ],
    'mood': [
        "happy", "happy", "happy", "happy", "happy",
        "sad", "sad", "sad", "sad", "sad",
        "devotional", "devotional", "devotional", "devotional", "devotional",
        "nervous", "nervous", "nervous", "nervous", "nervous"
    ]
}

df = pd.DataFrame(data)

# ✅ Use TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

model = MultinomialNB()
model.fit(X, df['mood'])

# ✅ Save model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickle.dump(model, open(os.path.join(BASE_DIR, 'mood_model.pkl'), 'wb'))
pickle.dump(vectorizer, open(os.path.join(BASE_DIR, 'mood_vectorizer.pkl'), 'wb'))

print("✅ Model retrained with improved dataset.")
