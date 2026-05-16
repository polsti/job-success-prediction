# Job Success Prediction

A personal ML project to predict whether a job application is likely to be successful,
based on job post features and how well my profile matches.

## Goal

Binary classification: **Will I get an offer? (Yes / No)**

## Features used

- Years of experience
- Tech stack match (how many required skills I have)
- Country / location
- Job type (remote, hybrid, onsite)
- Company size

## Project structure

```
data/           → datasets (CSV files)
notebooks/      → Jupyter notebooks for exploration
src/            → Python scripts
  preprocessing.py  → load and clean data
  train_model.py    → train and save the model
  evaluate.py       → measure model accuracy
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Add your dataset to `data/` as `jobs.csv`

3. Run the pipeline:
   ```
   python src/preprocessing.py
   python src/train_model.py
   python src/evaluate.py
   ```

## Stack

- Python
- pandas
- scikit-learn
