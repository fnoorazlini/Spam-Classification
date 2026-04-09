from flask import Flask, request, render_template
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
best_model = joblib.load(r"C:\Users\User\Desktop\SPAM\SpamClassification\best_model_80_20\best_model_80_20.pkl")

# Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message_text = ''
    result_class = ''

    if request.method == 'POST':
        action = request.form.get('submit_action')

        if action == 'predict':
            user_input = request.form['message']
            message_text = user_input

            cleaned_text = preprocess_text(user_input)
            vectorized = tfidf.transform([cleaned_text])

            pred = best_model.predict(vectorized)[0]
            probas = best_model.predict_proba(vectorized)[0]
            confidence = round(probas[pred] * 100, 2)

            if pred == 1:
                prediction = f"It is a Spam message ({confidence:.2f}% )."
                result_class = 'spam'
            else:
                prediction = f"It is a Ham message ({confidence:.2f}% )."
                result_class = 'ham'

        elif action == 'clear':
            message_text = ''
            prediction = None
            result_class = ''

    return render_template('spam.html', prediction=prediction, message_text=message_text, result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)
