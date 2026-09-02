from flask import Flask, request, render_template_string
import pickle
import re
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# NLTK DATA
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =========================================================
# LOAD MODEL AND TF-IDF
# =========================================================

with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)


# =========================================================
# TEXT PREPROCESSING
# =========================================================

stop_words = set(stopwords.words("english"))

# Keep important negation words
negation_words = {"no", "not", "nor", "never"}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove emojis / non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Keep only alphabets and spaces
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

    # Join tokens
    text = " ".join(tokens)

    return text


# =========================================================
# HTML + CSS
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

    <title>Emotion Classifier</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
        }

        body {

            margin: 0;

            min-height: 100vh;

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #0f172a,
                    #1e1b4b,
                    #312e81
                );

            display: flex;

            justify-content: center;

            align-items: center;

            padding: 30px;

            color: #ffffff;
        }


        .container {

            width: 100%;

            max-width: 750px;

            background: rgba(255,255,255,0.10);

            backdrop-filter: blur(18px);

            -webkit-backdrop-filter: blur(18px);

            border: 1px solid rgba(255,255,255,0.15);

            border-radius: 24px;

            padding: 40px;

            box-shadow:
                0 25px 60px rgba(0,0,0,0.35);
        }


        .header {

            text-align: center;

            margin-bottom: 30px;
        }


        .emoji {

            font-size: 55px;

            margin-bottom: 10px;
        }


        h1 {

            margin: 0;

            font-size: 34px;

            font-weight: 700;
        }


        .subtitle {

            margin-top: 10px;

            color: #cbd5e1;

            font-size: 16px;
        }


        textarea {

            width: 100%;

            height: 170px;

            resize: none;

            border: none;

            outline: none;

            border-radius: 16px;

            padding: 18px;

            font-size: 17px;

            font-family: inherit;

            background: rgba(255,255,255,0.95);

            color: #1e293b;

            box-shadow:
                inset 0 2px 8px rgba(0,0,0,0.08);
        }


        textarea::placeholder {

            color: #64748b;
        }


        .button {

            width: 100%;

            margin-top: 18px;

            padding: 15px;

            border: none;

            border-radius: 14px;

            font-size: 17px;

            font-weight: 600;

            cursor: pointer;

            color: white;

            background:
                linear-gradient(
                    90deg,
                    #6366f1,
                    #8b5cf6
                );

            transition: 0.25s;

            box-shadow:
                0 8px 20px rgba(99,102,241,0.35);
        }


        .button:hover {

            transform: translateY(-2px);

            box-shadow:
                0 12px 25px rgba(99,102,241,0.45);
        }


        .result {

            margin-top: 28px;

            padding: 25px;

            text-align: center;

            border-radius: 18px;

            background: rgba(255,255,255,0.10);

            border: 1px solid rgba(255,255,255,0.15);
        }


        .result-title {

            color: #cbd5e1;

            font-size: 15px;

            margin-bottom: 8px;
        }


        .prediction {

            font-size: 35px;

            font-weight: 700;

            text-transform: capitalize;

            margin-bottom: 12px;

            color: #a5b4fc;
        }


        .confidence {

            font-size: 18px;

            color: #e2e8f0;
        }


        .bar-container {

            width: 100%;

            height: 10px;

            background: rgba(255,255,255,0.15);

            border-radius: 10px;

            margin-top: 15px;

            overflow: hidden;
        }


        .bar {

            height: 100%;

            border-radius: 10px;

            background:
                linear-gradient(
                    90deg,
                    #818cf8,
                    #c084fc
                );
        }


        .info {

            margin-top: 25px;

            text-align: center;

            color: #94a3b8;

            font-size: 13px;
        }


        .classes {

            margin-top: 15px;

            display: flex;

            flex-wrap: wrap;

            justify-content: center;

            gap: 8px;
        }


        .class-tag {

            padding: 6px 12px;

            border-radius: 20px;

            background: rgba(255,255,255,0.08);

            color: #cbd5e1;

            font-size: 12px;
        }


        @media (max-width: 600px) {

            body {
                padding: 15px;
            }

            .container {
                padding: 25px;
            }

            h1 {
                font-size: 28px;
            }

            .emoji {
                font-size: 45px;
            }

        }

    </style>

</head>


<body>


<div class="container">


    <div class="header">

        <div class="emoji">🧠💭</div>

        <h1>Emotion Classifier</h1>

        <div class="subtitle">
            Enter a sentence and discover the emotion behind it.
        </div>

    </div>


    <form method="POST">

        <textarea
            name="text"
            placeholder="Example: I am extremely happy today..."
            required>{{ text }}</textarea>


        <button class="button" type="submit">

            🔮 Predict Emotion

        </button>

    </form>


    {% if prediction %}

    <div class="result">

        <div class="result-title">
            Predicted Emotion
        </div>


        <div class="prediction">

            {{ emotion_emoji }} {{ prediction }}

        </div>


        <div class="confidence">

            Confidence: {{ confidence }}%

        </div>


        <div class="bar-container">

            <div
                class="bar"
                style="width: {{ confidence }}%;">
            </div>

        </div>

    </div>

    {% endif %}


    <div class="info">

        Powered by TF-IDF + Logistic Regression

        <div class="classes">

            <span class="class-tag">😡 Anger</span>

            <span class="class-tag">😨 Fear</span>

            <span class="class-tag">😊 Joy</span>

            <span class="class-tag">❤️ Love</span>

            <span class="class-tag">😢 Sadness</span>

            <span class="class-tag">😮 Surprise</span>

        </div>

    </div>


</div>


</body>

</html>

"""


# =========================================================
# EMOTION EMOJIS
# =========================================================

emotion_emojis = {

    "anger": "😡",

    "fear": "😨",

    "joy": "😊",

    "love": "❤️",

    "sadness": "😢",

    "surprise": "😮"

}


# =========================================================
# HOME / PREDICTION
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    confidence = None

    text = ""

    emotion_emoji = "🧠"


    if request.method == "POST":

        text = request.form.get("text", "").strip()


        if text:

            # Same preprocessing used during training
            cleaned_text = clean_text(text)


            # IMPORTANT:
            # Only transform using the already-fitted TF-IDF
            text_vector = tfidf.transform([cleaned_text])


            # Prediction
            prediction = model.predict(text_vector)[0]


            # Probability
            probabilities = model.predict_proba(text_vector)[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )


            emotion_emoji = emotion_emojis.get(
                prediction,
                "🧠"
            )


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        text=text,

        emotion_emoji=emotion_emoji

    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
