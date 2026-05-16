import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_data, clean_data, encode_features, split_features_and_target


def train(filepath):
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)

    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy:  {test_accuracy:.2f}")

    # Save the trained model so evaluate.py can load it
    with open("data/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to data/model.pkl")
    return model, X_test, y_test


if __name__ == "__main__":
    train("data/jobs_processed.csv")
