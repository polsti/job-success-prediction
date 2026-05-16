import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ── STEP 1: Load the processed data ───────────────────────────────────────────
# We load the cleaned file created by preprocessing.py — not the raw survey.

def load_processed(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ── STEP 2: Separate features from target ─────────────────────────────────────
# X = everything the model uses to make a prediction (the input)
# y = what we want to predict (the answer)
#
# Think of it like a student: X is the exam question, y is the correct answer.
# During training, the model sees both. During testing, it only sees X.

def split_X_y(df):
    X = df.drop(columns=["got_offer"])
    y = df["got_offer"]
    return X, y


# ── STEP 3: Split into training and test sets ─────────────────────────────────
# We hide 20% of the data from the model during training.
# After training, we test on those hidden rows to see if the model truly learned.
#
# stratify=y → makes sure both sets have the same ratio of 0s and 1s.
# This is important because our data is imbalanced (93% employed).
# Without it, the test set might end up with very few "not employed" examples.

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% goes to test set
        random_state=42,    # fixed seed → same split every time you run it
        stratify=y          # keeps class ratio balanced in both sets
    )
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows:     {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── STEP 4: Train the model ────────────────────────────────────────────────────
# RandomForestClassifier builds many decision trees and combines their votes.
#
# class_weight="balanced" → the key fix for our imbalanced data.
# It tells the model: "pay more attention to the rare class (not employed)."
# Without it, the model would just predict "employed" for everyone and stop trying.
#
# n_estimators=100 → build 100 trees. More trees = more stable predictions.
# random_state=42  → so results are reproducible.

def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model trained.")
    return model


# ── STEP 5: Evaluate on the test set ──────────────────────────────────────────
# We ask the model to predict on rows it has never seen before.
# classification_report shows:
#   - precision: when the model says "employed", how often is it right?
#   - recall: of all actually employed people, how many did it catch?
#   - f1-score: a balance between precision and recall

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\n── Evaluation on test set ──")
    print(classification_report(y_test, predictions, target_names=["not employed", "employed"]))


# ── STEP 6: Save the model ────────────────────────────────────────────────────
# We save the trained model to a file so we can load it later in evaluate.py
# without having to retrain from scratch every time.
# pickle is Python's built-in way to save any object to a file.

def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


# ── Run everything ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_processed("data/processed.csv")
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, "data/model.pkl")
