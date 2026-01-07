"""
Here’s a step-by-step guide (with working Python code) to build a simple sports betting forecasting app using machine learning. This will:

✅ load historical match data
✅ train a model to predict outcomes
✅ provide predictions in a simple UI (terminal or Flask web app)

You can later improve it with live data, better models, and real odds.

⸻

⭐ 1) What This App Does

Given historical match results (e.g., soccer), the app:

📊 Extracts features (like team strength, goals, form)
🤖 Trains a prediction model (e.g., Logistic Regression)
📈 Predicts outcomes for future games (Home Win / Draw / Away Win)

⸻

🧾 2) Data You Need

You need historical match results with columns like:

Date	HomeTeam	AwayTeam	HomeGoals	AwayGoals	Result

Result could be 'H' (home win), 'D' (draw), 'A' (away win).

Example dataset: EPL 2010–2025 CSV

Save it as matches.csv.

⸻

🧠 3) Python Code (Training + Predicting)

Install first:

pip install pandas scikit-learn flask

📌 model.py — Train & Save Model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("matches.csv")

# Feature engineering
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

X = df[["HomeTeam", "AwayTeam", "GoalDiff"]]
y = df["Result"]

cat_features = ["HomeTeam", "AwayTeam"]
num_features = ["GoalDiff"]

preprocess = ColumnTransformer(
    transformers=[
        ("cats", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=200))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model score:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "betting_model.pkl")

"""
🧪 4) Predicting Matches

📌 predict.py
"""

import joblib
import pandas as pd

model = joblib.load("betting_model.pkl")

def predict_match(home, away, hd=0):
    df = pd.DataFrame([{"HomeTeam": home, "AwayTeam": away, "GoalDiff": hd}])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob

if __name__ == "__main__":
    home = input("Home Team: ")
    away = input("Away Team: ")
    prediction, probabilities = predict_match(home, away)
    print(f"Predicted: {prediction}")
    print("Probabilities:", probabilities)

"""
You run:

python predict.py

And enter team names to get a prediction.

⸻

🌐 5) Optional: Make It a Web App (Flask)

📌 app.py
"""

from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("betting_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        home_team = request.form["home"]
        away_team = request.form["away"]
        df = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team, "GoalDiff": 0}])
        pred = model.predict(df)[0]
        result = f"Prediction: {pred}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

# templates/index.html
"""
<!doctype html>
<html>
  <body>
    <h1>Betting Prediction</h1>
    <form method="POST">
      Home Team: <input name="home" /><br>
      Away Team: <input name="away" /><br>
      <button type="submit">Predict</button>
    </form>
    <div>{{ result }}</div>
  </body>
</html>

Run:

python app.py

Then open http://localhost:5000


🚀 6) Ways to Improve

🔹 Use more features
• Recent form (last 5 matches)
• Home/Away goals per season
• Team Elo ratings

🔹 Use better models
• Random Forest
• XGBoost / LightGBM
• Neural Networks

🔹 Add Odds Comparison
• Use bookmaker odds as features for probability calibration

🔹 Deploy to cloud
• Heroku / Railway / Render

⸻

⚠️ Important Notes

⚖️ Betting models give probabilities, not certainties
💸 Always gamble responsibly
📈 Better models = better insights, but no guarantees

⸻

If you want, tell me:

➡ Your dataset format
➡ Sport (soccer/cricket/tennis etc.)
➡ Whether you want live odds integration

…and I’ll tailor the app further!
"""