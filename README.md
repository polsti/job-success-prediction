# Job Success Prediction

A beginner ML project that predicts whether a developer is likely to be employed,
based on their experience, skills, education, location, and work preferences.

Built using the **Stack Overflow Developer Survey 2025** (49,000+ real responses).

---

## Goal

Binary classification: **Employed (1) or Not employed (0)**

The model learns from real survey data — what profile of developer tends to be employed?

---

## Dataset

**Source:** Stack Overflow Developer Survey 2025
**Download:** survey.stackoverflow.co → Data (CSV)
**Save as:** `data/survey_results.csv`

> The dataset is not committed to this repo (134MB). Download it manually using the link above.

---

## Features used

| Feature | Description |
|---|---|
| `WorkExp` | Years of professional work experience |
| `num_skills` | Number of programming languages known |
| `EdLevel` | Education level (bachelor / master / phd / other) |
| `Country` | Country of residence |
| `RemoteWork` | Remote / hybrid / in-person |
| `OrgSize` | Company size (small / medium / large) |

**Target:** `got_offer` — 1 if employed, 0 if not employed

---

## Results

Model: **Random Forest Classifier** with `class_weight="balanced"`

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not employed | 0.42 | 0.86 | 0.56 |
| Employed | 0.99 | 0.92 | 0.95 |
| **Overall accuracy** | | | **0.92** |

Key finding: the strongest predictor was **missing company data** — people who left
OrgSize and RemoteWork blank were much more likely to be unemployed. This is called
*informative missingness*: the absence of an answer is itself a signal.

---

## Project structure

```
job-success-prediction/
│
├── data/                   ← datasets go here (not committed to git)
├── plots/                  ← saved visualisation images
├── src/
│   ├── preprocessing.py    ← load raw survey, clean, encode, save processed.csv
│   ├── train_model.py      ← train Random Forest, save model.pkl
│   ├── evaluate.py         ← load model, confusion matrix, feature importance
│   └── visualize.py        ← generate and save 4 plots as PNG
└── requirements.txt
```

---

## How to run

**1. Clone the repo and set up the environment**
```bash
git clone https://github.com/polsti/job-success-prediction.git
cd job-success-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download the dataset**

Go to `survey.stackoverflow.co`, click **Data (CSV)**, save the file as:
```
data/survey_results.csv
```

**3. Run the pipeline**
```bash
python src/preprocessing.py   # cleans data → saves data/processed.csv
python src/train_model.py     # trains model → saves data/model.pkl
python src/evaluate.py        # prints evaluation report
python src/visualize.py       # saves 4 plots to plots/
```

---

## Visualisations

**Class distribution** — shows the 93% / 7% imbalance in the dataset

**Confusion matrix** — where the model makes mistakes

**Feature importance** — which columns drove the predictions most

**Precision vs Recall** — how performance differs between the two classes

---

## Stack

- Python 3
- pandas
- scikit-learn
- matplotlib
- seaborn
