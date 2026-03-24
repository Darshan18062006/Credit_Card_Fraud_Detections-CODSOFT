import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    train_path = os.path.join(DATA_DIR, "fraudTrain.csv")
    df = pd.read_csv(train_path)

    # Use only the columns needed; example columns from this dataset:
    # trans_date_trans_time, cc_num, merchant, category, amt, gender, ... is_fraud
    # we'll keep only numeric and properly encoded ones
    features = [
        "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"
    ]
    X = df[features]
    y = df["is_fraud"]

    # Drop rows with NaN (optional; you can impute instead)
    X = X.dropna()
    y = y.loc[X.index]

    return X, y

def preprocess_and_train():
    X, y = load_data()

    # Encode categorical columns if needed (e.g., category, gender, etc.)
    # For simplicity here we assume `X` is numeric; if you add categorical
    # features, use LabelEncoder or OneHotEncoder on those columns.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to experiment with
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        print(f"{name} test accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

        # Save each model + scaler
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        if name == best_name:
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(f"Best model: {best_name} (acc: {best_score:.4f})")

    # Save best model name
    with open(os.path.join(MODEL_DIR, "best_model.txt"), "w") as f:
        f.write(best_name)

    # Optional: print detailed report for best model
    y_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    preprocess_and_train()
