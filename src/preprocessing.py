import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    # Drop rows where the target column is missing
    df = df.dropna(subset=["got_offer"])

    # Fill missing numeric values with the column median
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing text values with "unknown"
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].fillna("unknown")

    return df


def encode_features(df):
    # Convert text columns (like country, job_type) into numbers
    text_cols = df.select_dtypes(include="object").columns.tolist()

    # Don't encode the target column
    if "got_offer" in text_cols:
        text_cols.remove("got_offer")

    df = pd.get_dummies(df, columns=text_cols)
    return df


def split_features_and_target(df):
    X = df.drop(columns=["got_offer"])
    y = df["got_offer"]
    return X, y


if __name__ == "__main__":
    df = load_data("data/jobs.csv")
    print(f"Loaded {len(df)} rows")

    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")

    df = encode_features(df)
    print(f"Columns after encoding: {list(df.columns)}")

    df.to_csv("data/jobs_processed.csv", index=False)
    print("Saved to data/jobs_processed.csv")
