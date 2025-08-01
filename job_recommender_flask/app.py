from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import torch
app = Flask(__name__)
print("🔥 app.py started running...")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model and data



# Safely load model on CPU even if it was saved on CUDA
with open('bert_model_new.pkl', 'rb') as f:
    model = pickle.load(f)



df = pd.read_csv('processed_job_data.csv')  # ✅ Back to CSV

  

# Preprocess and embed combined text
df['processed_text'] = df['combined_text'].apply(preprocess_text)


if os.path.exists("embedded_jobs.pkl"):
    df = pd.read_pickle("embedded_jobs.pkl")
    print("✅ Loaded existing embeddings.")
else:
    print("⏳ Computing embeddings...")

    df = df.dropna(subset=['processed_text'])

    # Batch encode
    df['embedding'] = model.encode(df['processed_text'].tolist(), show_progress_bar=True).tolist()

    # Save for reuse
    df.to_pickle("embedded_jobs.pkl")
    print("✅ Embeddings saved.")



# Recommendation function
def get_recommendations(user_input):
    processed_input = preprocess_text(user_input)
    user_embedding = model.encode(processed_input)

    # Compute cosine similarity
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])

    top_matches = df.sort_values(by='similarity', ascending=False).head(5)

    result = []
    for _, row in top_matches.iterrows():
        rec = {
            'title': row.get('jobtitle', ''),
            'description': row.get('jobdescription', ''),
            'skills': row.get('skills', ''),
            'education': row.get('education', ''),
            'match': round(row['similarity'] * 100, 2)
        }
        result.append(rec)
    return result

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        profile = request.form.get('profile', '')
        if profile.strip():
            recommendations = get_recommendations(profile)
    return render_template('index.html', recommendations=recommendations)
    
if __name__== '__main__':
    print("✅ Running Flask app...")
    app.run(debug=True)