"""Train and save a small demo classifier. Run once before starting the API."""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = load_iris()
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42)),
])
model.fit(data.data, data.target)
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
print("Class mapping:", dict(enumerate(data.target_names)))
