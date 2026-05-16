import pandas as pd

#load dataset
# only data that is needed for training the model is loaded
COLUMNS =[
    "Employment",               # target
    "WorkExp",                  # years of experience
    "EdLevel",                  # education level
    "Country",                  # country of residence
    "RemoteWork",               # actual column name in the CSV (not WorkType)
    "OrgSize",                  # company size
    "LanguageHaveWorkedWith",   # actual column name in the CSV (not Skills)
]
def load_data(path):
    df = pd.read_csv(path, usecols=COLUMNS)
    print(f"Dataset loaded: {len(df)} rows")
    return df

# target column 
# Employed = 1 (success), Not employed = 0 (no success)  
# drop student, retured, freelancers
def create_target(df):
    keep = ["Employed", "Not employed"]  # exact values from the CSV
    df = df[df["Employment"].isin(keep)].copy()
    df["got_offer"] = (df["Employment"] == "Employed").astype(int)
    df = df.drop(columns=["Employment"])  # must save the result back to df
    print(f"After filtering: {len(df)} rows")
    print(f"Got offer: {df['got_offer'].mean():.1%} of rows")
    return df

# deal with messy values 
# shorten long texts 
def simplify_education(value):
    if pd.isna(value):
        return "other"
    v = value.lower()
    if "bachelor" in v:
        return "bachelor"
    elif "master" in v:
        return "master"
    elif "phd" in v or "doctor" in v:
        return "phd"
    elif "secondary" in v or "high school" in v:
        return "no degree"
    return "other"

def simplify_remote(value):
    if pd.isna(value):
        return "unknown"
    v = value.lower()
    if "remote" in v and "hybrid" not in v:
        return "remote"
    elif "hybrid" in v:
        return "hybrid"
    elif "in-person" in v or "in person" in v:
        return "in_person"
    return "flexible"

def simplify_org_size(value):
    if pd.isna(value):
        return "unknown"
    v = value.lower()
    if "less than 20" in v or "just me" in v or "freelancer" in v:
        return "small"
    elif "20 to 99" in v or "100 to 499" in v:
        return "medium"
    elif "500" in v or "1,000" in v or "5,000" in v or "10,000" in v:
        return "large"
    return "unknown"

def count_languages(value):
    if pd.isna(value):
        return 0
    return len(value.split(";"))  # data uses semicolons: "Python;SQL;JavaScript"

def simplify_columns(df):
    df["EdLevel"]   = df["EdLevel"].apply(simplify_education)
    df["RemoteWork"] = df["RemoteWork"].apply(simplify_remote)
    df["OrgSize"]   = df["OrgSize"].apply(simplify_org_size)
    df["num_skills"] = df["LanguageHaveWorkedWith"].apply(count_languages)
    df = df.drop(columns=["LanguageHaveWorkedWith"])  # replaced by num_skills
    return df

# handle missing values
def handle_missing(df):
    df["WorkExp"] = df["WorkExp"].fillna(0) # assume no experience if missing
    df["Country"] = df["Country"].fillna("unknown")
    return df

# encode text columns
# ml models work only with numbers 
# hot encoding turns each text category into its own 0/1 col
def encode(df):
    text_cols = ["EdLevel", "Country", "RemoteWork", "OrgSize"]
    df = pd.get_dummies(df, columns=text_cols)
    return df


# python src/preprocessing.py

if __name__ == "__main__":
    df = load_data("data/survey_results.csv")
    df = create_target(df)
    df = simplify_columns(df)
    df = handle_missing(df)
    df = encode(df)

    df.to_csv("data/processed.csv", index=False)
    print(f"\nSaved to data/processed.csv")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"First few columns: {list(df.columns[:6])}")
