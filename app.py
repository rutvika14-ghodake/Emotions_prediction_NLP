from flask import Flask, request, render_template_string
import pickle
import re

app = Flask(__name__)

# --------------------------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------------------------

with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)

    # Remove emojis and non-English characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# HTML + CSS
# --------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>EmotionAI | Emotion Detector</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;

            min-height: 100vh;

            background:
                radial-gradient(circle at top left, #312e81 0%, transparent 35%),
                radial-gradient(circle at bottom right, #0f766e 0%, transparent 35%),
                #0f172a;

            color: #f8fafc;

            display: flex;
            align-items: center;
            justify-content: center;

            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 850px;
        }

        .card {
            background: rgba(15, 23, 42, 0.82);

            border: 1px solid rgba(255, 255, 255, 0.12);

            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);

            border-radius: 28px;

            padding: 45px;

            box-shadow:
                0 25px 60px rgba(0, 0, 0, 0.35);
        }

        .logo {
            width: 65px;
            height: 65px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 18px;

            background: linear-gradient(
                135deg,
                #8b5cf6,
                #06b6d4
            );

            font-size: 30px;

            margin-bottom: 20px;

            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
        }

        h1 {
            font-size: 42px;
            line-height: 1.1;

            margin-bottom: 12px;

            background: linear-gradient(
                90deg,
                #c4b5fd,
                #67e8f9
            );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: #94a3b8;

            font-size: 17px;

            line-height: 1.6;

            margin-bottom: 32px;
        }

        textarea {
            width: 100%;

            min-height: 180px;

            resize: vertical;

            background: rgba(30, 41, 59, 0.75);

            border: 1px solid #334155;

            border-radius: 18px;

            padding: 20px;

            color: #f8fafc;

            font-size: 16px;

            font-family: inherit;

            outline: none;

            transition: 0.25s;
        }

        textarea::placeholder {
            color: #64748b;
        }

        textarea:focus {
            border-color: #8b5cf6;

            box-shadow:
                0 0 0 4px rgba(139, 92, 246, 0.12);
        }

        .button {
            width: 100%;

            margin-top: 18px;

            padding: 16px;

            border: none;

            border-radius: 15px;

            cursor: pointer;

            font-size: 16px;

            font-weight: 700;

            color: white;

            background: linear-gradient(
                135deg,
                #7c3aed,
                #0891b2
            );

            transition: all 0.25s ease;

            box-shadow:
                0 10px 25px rgba(124, 58, 237, 0.25);
        }

        .button:hover {
            transform: translateY(-2px);

            box-shadow:
                0 15px 30px rgba(124, 58, 237, 0.35);
        }

        .button:active {
            transform: translateY(0);
        }

        .result {
            margin-top: 30px;

            padding: 25px;

            border-radius: 20px;

            background: rgba(30, 41, 59, 0.75);

            border: 1px solid rgba(255,255,255,0.1);

            text-align: center;
        }

        .result-label {
            color: #94a3b8;

            font-size: 14px;

            text-transform: uppercase;

            letter-spacing: 2px;

            margin-bottom: 10px;
        }

        .emotion {
            font-size: 36px;

            font-weight: 800;

            background: linear-gradient(
                90deg,
                #c4b5fd,
                #67e8f9
            );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .confidence {
            margin-top: 10px;

            color: #94a3b8;

            font-size: 14px;
        }

        .info {
            display: flex;

            justify-content: center;

            gap: 12px;

            flex-wrap: wrap;

            margin-top: 28px;
        }

        .tag {
            padding: 8px 14px;

            border-radius: 999px;

            background: rgba(51, 65, 85, 0.55);

            color: #cbd5e1;

            font-size: 13px;

            border: 1px solid rgba(255,255,255,0.08);
        }

        .error {
            margin-top: 20px;

            padding: 15px;

            border-radius: 12px;

            background: rgba(127, 29, 29, 0.25);

            color: #fca5a5;

            border: 1px solid rgba(248, 113, 113, 0.2);

            text-align: center;
        }

        footer {
            text-align: center;

            margin-top: 25px;

            color: #64748b;

            font-size: 13px;
        }

        @media (max-width: 600px) {

            body {
                padding: 15px;
            }

            .card {
                padding: 28px 20px;

                border-radius: 22px;
            }

            h1 {
                font-size: 32px;
            }

            .subtitle {
                font-size: 15px;
            }

            textarea {
                min-height: 150px;
            }

            .emotion {
                font-size: 30px;
            }
        }

    </style>
</head>


<body>

<div class="container">

    <div class="card">

        <div class="logo">
            🧠
        </div>

        <h1>EmotionAI</h1>

        <p class="subtitle">
            Discover the emotion hidden in your text using
            Machine Learning and Natural Language Processing.
        </p>


        <form method="POST">

            <textarea
                name="text"
                placeholder="Type something like: 
I am so happy today because I got my dream job!"
                required
            >{{ text }}</textarea>

            <button class="button" type="submit">
                ✨ Detect Emotion
            </button>

        </form>


        {% if prediction %}

        <div class="result">

            <div class="result-label">
                Detected Emotion
            </div>

            <div class="emotion">
                {{ prediction }}
            </div>

            {% if confidence %}
            <div class="confidence">
                Model confidence: {{ confidence }}%
            </div>
            {% endif %}

        </div>

        {% endif %}


        {% if error %}

        <div class="error">
            {{ error }}
        </div>

        {% endif %}


        <div class="info">

            <span class="tag">TF-IDF</span>

            <span class="tag">Logistic Regression</span>

            <span class="tag">NLP</span>

            <span class="tag">6 Emotions</span>

        </div>

    </div>


    <footer>
        Built with Python • Flask • Scikit-learn
    </footer>

</div>

</body>

</html>
"""


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if not text:

            error = "Please enter some text."

        else:

            try:

                # Clean user input
                cleaned_text = clean_text(text)

                # Convert text into TF-IDF features
                text_vector = tfidf.transform([cleaned_text])

                # Prediction
                prediction = model.predict(text_vector)[0]

                # Probability / confidence
                probabilities = model.predict_proba(text_vector)[0]

                confidence = round(max(probabilities) * 100, 2)

                # Capitalize emotion
                prediction = prediction.capitalize()

            except Exception as e:

                error = "Something went wrong while processing your text."

                print("ERROR:", e)


    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        error=error,
        text=text
    )


# --------------------------------------------------
# Render / Production Server
# --------------------------------------------------

if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
