import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# STEP 1: Load the saved model 
# load the model that train_model.py already saved.
# This is the same pattern used in real products:
# train once (expensive), load and use many times (cheap).

def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


#  STEP 2: Recreate the same test set 
# exact same random_state=42 and stratify=y as in train_model.py
# This guarantees the identical 20% test rows — not rows the model trained on.

def get_test_set(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["got_offer"])
    y = df["got_offer"]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_test, y_test


# STEP 3: Confusion matrix 
# Shows exactly where the model is making mistakes.
#
# Layout:
#                     Predicted NOT employed   Predicted EMPLOYED
# Actually NOT employed       TN                      FP
# Actually EMPLOYED           FN                      TP
#
# TN = correctly said "not employed"
# TP = correctly said "employed"
# FP = said "not employed" but they were actually employed  (false alarm)
# FN = said "employed" but they were actually not employed  (missed)

def show_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    print("\n── Confusion Matrix ──────────────────────────────")
    print(f"                  Predicted: NOT   Predicted: EMPLOYED")
    print(f"Actually: NOT         {tn:<6}        {fp}")
    print(f"Actually: EMPLOYED    {fn:<6}        {tp}")
    print()
    print(f"Correctly identified NOT employed (TN): {tn}")
    print(f"Correctly identified EMPLOYED (TP):     {tp}")
    print(f"False alarms — said NOT but actually employed (FP): {fp}")
    print(f"Missed — said EMPLOYED but actually not (FN):       {fn}")


# STEP 4: Feature importance 
# Random Forest tracks how useful each column was for making decisions.
# A high importance = this column helped the model split data correctly.
# A low importance = this column barely mattered.
def show_feature_importance(model, X_test, top_n=10):
    importances = model.feature_importances_
    feature_names = X_test.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\n── Top {top_n} most important features ───────────────────")
    for i, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")


#  STEP 5: Single example prediction 
# pick one real row from the test set and show what the model predicted.
# This makes the model feel concrete — not just numbers on a report.

def show_example(model, X_test, y_test):
    example = X_test.iloc[0:1]  # first row, kept as a DataFrame
    actual    = y_test.iloc[0]
    predicted = model.predict(example)[0]
    probability = model.predict_proba(example)[0]

    labels = {0: "NOT employed", 1: "EMPLOYED"}
    print("\n── Single prediction example ─────────────────────")
    print(f"  WorkExp:    {example['WorkExp'].values[0]}")
    print(f"  num_skills: {example['num_skills'].values[0]}")
    print(f"  Actual:     {labels[actual]}")
    print(f"  Predicted:  {labels[predicted]}")
    print(f"  Confidence: {max(probability):.1%}")

if __name__ == "__main__":
    model              = load_model("data/model.pkl")
    X_test, y_test     = get_test_set("data/processed.csv")
    predictions        = model.predict(X_test)

    print("\n── Classification Report ─────────────────────────")
    print(classification_report(y_test, predictions, target_names=["not employed", "employed"]))

    show_confusion_matrix(y_test, predictions)
    show_feature_importance(model, X_test, top_n=10)
    show_example(model, X_test, y_test)
