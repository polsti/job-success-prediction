import pandas as pd
import numpy as np

np.random.seed(42)
N = 4500

countries = ["Lithuania", "Spain", "Germany", "USA", "Poland", "Netherlands", "UK", "Ukraine"]
job_types = ["remote", "hybrid", "onsite"]
education = ["self-taught", "bootcamp", "bachelor", "master", "phd"]
company_sizes = ["small", "medium", "large"]

df = pd.DataFrame({
    "years_experience":  np.random.randint(0, 15, N),
    "skills_match":      np.random.randint(0, 11, N),   # how many required skills you have (out of 10)
    "country":           np.random.choice(countries, N),
    "job_type":          np.random.choice(job_types, N),
    "education":         np.random.choice(education, N),
    "company_size":      np.random.choice(company_sizes, N),
    "num_applications":  np.random.randint(1, 50, N),   # how many jobs applied to
})

# Build a realistic target: more experience + skills → higher chance of offer
score = (
    df["years_experience"] * 0.05 +
    df["skills_match"] * 0.08 +
    (df["education"] == "master").astype(int) * 0.1 +
    (df["education"] == "phd").astype(int) * 0.12 +
    (df["job_type"] == "remote").astype(int) * 0.05 +
    np.random.normal(0, 0.15, N)   # some randomness — not everything is predictable
)

df["got_offer"] = (score > score.median()).astype(int)

df.to_csv("jobs.csv", index=False)
print(f"Dataset created: {len(df)} rows, {df.columns.tolist()}")
print(f"\nOffer rate: {df['got_offer'].mean():.1%}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
