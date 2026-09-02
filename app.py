from flask import Flask, request, render_template_string
import pickle
import re
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# APP
# =========================================================

app = Flask(__name__)


# =========================================================
# NLTK
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =========================================================
# LOAD MODEL
# =========================================================

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


# =========================================================
# PREPROCESSING
# =========================================================

stop_words = set(stopwords.words("english"))

# Keep negation words
negation_words = {"no", "not", "nor", "never"}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove emojis / non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# HTML
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Emotion Predictor</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #312e81,
            #581c87
        );

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 25px;

}


.card {

    width: 100%;

    max-width: 720px;

    padding: 40px;

    border-radius: 25px;

    background: rgba(255,255,255,0.10);

    border: 1px solid rgba(255,255,255,0.20);

    backdrop-filter: blur(15px);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.40);

    color: white;

}


.header {

    text-align: center;

    margin-bottom: 30px;

}


.icon {

    font-size: 55px;

}


h1 {

    margin: 10px 0;

    font-size: 35px;

}


.subtitle {

    color: #cbd5e1;

    font-size: 16px;

}


textarea {

    width: 100%;

    height: 170px;

    padding: 18px;

    border: none;

    border-radius: 15px;

    outline: none;

    resize: none;

    font-size: 17px;

    font-family: Arial, sans-serif;

    color: #1e293b;

}


textarea::placeholder {

    color: #64748b;

}


button {

    width: 100%;

    margin-top: 18px;

    padding: 16px;

    border: none;

    border-radius: 14px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #a855f7
        );

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;

}


button:hover {

    opacity: 0.9;

    transform: translateY(-2px);

}


.result {

    margin-top: 30px;

    padding: 25px;

    text-align: center;

    border-radius: 18px;

    background: rgba(255,255,255,0.10);

}


.result-title {

    color: #cbd5e1;

    font-size: 15px;

}


.prediction {

    margin-top: 10px;

    font-size: 36px;

    font-weight: bold;

    color: #c4b5fd;

    text-transform: capitalize;

}


.confidence {

    margin-top: 10px;

    font-size: 18px;

    color: #e2e8f0;

}


.bar-background {

    margin-top: 15px;

    width: 100%;

    height: 10px;

    background: rgba(255,255,255,0.15);

    border-radius: 10px;

}


.bar {

    height: 10px;

    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #c084fc
        );

}


.footer {

    margin-top: 25px;

    text-align: center;

    color: #94a3b8;

    font-size: 13px;

}


@media(max-width:600px) {

    .card {

        padding: 25px;

    }

    h1 {

        font-size: 28px;

    }

}

</style>

</head>


<body>


<div class="card">


<div class="header">

    <div class="icon">🧠✨</div>

    <h1>Emotion Predictor</h1>

    <div class="subtitle">

        Understand the emotion behind your text

    </div>

</div>


<form method="POST" action="/">

<textarea
    name="text"
    placeholder="Type something like: I am extremely happy today..."
    required>{{ text }}</textarea>


<button type="submit">

    🔮 Predict Emotion

</button>

</form>


{% if prediction %}

<div class="result">

    <div class="result-title">

        Predicted Emotion

    </div>


    <div class="prediction">

        {{ emoji }} {{ prediction }}

    </div>


    <div class="confidence">

        Confidence: {{ confidence }}%

    </div>


    <div class="bar-background">

        <div
            class="bar"
            style="width: {{ confidence }}%;">
        </div>

    </div>

</div>

{% endif %}


<div class="footer">

    Powered by TF-IDF + Logistic Regression

</div>


</div>


</body>

</html>

"""


# =========================================================
# EMOTION EMOJIS
# =========================================================

emojis = {

    "anger": "😡",

    "fear": "😨",

    "joy": "😊",

    "love": "❤️",

    "sadness": "😢",

    "surprise": "😮"

}


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    text = ""
    emoji = "🧠"


    if request.method == "POST":

        text = request.form.get("text", "").strip()


        if text:

            cleaned_text = clean_text(text)

            # IMPORTANT:
            # Do NOT fit the TF-IDF again.
            vector = tfidf.transform([cleaned_text])

            prediction = model.predict(vector)[0]

            probabilities = model.predict_proba(vector)[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            emoji = emojis.get(
                prediction,
                "🧠"
            )


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        text=text,

        emoji=emoji

    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return "Emotion Classifier is running!"


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
