import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#  Setup 
# Apply a clean visual style to all plots.
# "whitegrid" adds light horizontal lines so values are easier to read.
sns.set_style("whitegrid")


def load_data_and_model():
    df    = pd.read_csv("data/processed.csv")
    X     = df.drop(columns=["got_offer"])
    y     = df["got_offer"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    return df, X_test, y_test, predictions, model


#  PLOT 1: Class distribution 
# Shows how many rows are "employed" vs "not employed" in the full dataset.
# This makes the imbalance visible — 93% vs 7% is very obvious on a bar chart.
# Knowing your class imbalance upfront is one of the first things you do in any ML project.

def plot_class_distribution(df):
    counts = df["got_offer"].map({1: "Employed", 0: "Not employed"}).value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4C9BE8", "#E8714C"], width=0.5)

    # Add the exact number on top of each bar
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{int(bar.get_height()):,}",
            ha="center", fontsize=11
        )

    ax.set_title("Class Distribution — how balanced is our data?", fontsize=13)
    ax.set_ylabel("Number of people")
    ax.set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    plt.savefig("plots/class_distribution.png", dpi=150)
    plt.close()
    print("Saved: plots/class_distribution.png")


#  PLOT 2: Confusion matrix heatmap 
# A heatmap shows the same confusion matrix as before but visually.
# Darker colour = more predictions in that cell.
# The ideal result: dark diagonal (TN and TP), light off-diagonal (FP and FN).
# If an off-diagonal cell is very dark, the model has a systematic mistake there.

def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    labels = ["Not employed", "Employed"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,         # write the numbers inside each cell
        fmt="d",            # format as integers (not 6.0, just 6)
        cmap="Blues",       # light-to-dark blue colour scale
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: plots/confusion_matrix.png")


#  PLOT 3: Feature importance 
# A horizontal bar chart of the top 10 most useful features.
# Longer bar = model relied on this feature more when making decisions.
# This is unique to tree-based models (Random Forest, XGBoost etc.).
# It tells you: if you could only keep a few columns, which would you pick?

def plot_feature_importance(model, X_test, top_n=10):
    importance_df = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#4C9BE8")

    ax.set_title(f"Top {top_n} Most Important Features", fontsize=13)
    ax.set_xlabel("Importance score")

    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: plots/feature_importance.png")


#  PLOT 4: Precision and Recall per class   
# Precision and Recall for each class side by side.
# This makes the imbalance in model performance visible:
# - "Employed" class: both bars will be high (model is confident and accurate)
# - "Not employed" class: recall will be high but precision low
#   → the model finds most unemployed people, but also raises too many false alarms
#
# Precision = of all predicted as class X, how many were actually X?
# Recall    = of all actual class X, how many did the model catch?

def plot_precision_recall(y_test, predictions):
    report = classification_report(
        y_test, predictions,
        target_names=["Not employed", "Employed"],
        output_dict=True    # returns a dictionary instead of a string
    )

    classes    = ["Not employed", "Employed"]
    precisions = [report[c]["precision"] for c in classes]
    recalls    = [report[c]["recall"]    for c in classes]

    x = range(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width/2 for i in x], precisions, width, label="Precision", color="#4C9BE8")
    ax.bar([i + width/2 for i in x], recalls,    width, label="Recall",    color="#E8714C")

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision vs Recall per Class", fontsize=13)
    ax.legend()

    # Add value labels on each bar
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        ax.text(i - width/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=10)
        ax.text(i + width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("plots/precision_recall.png", dpi=150)
    plt.close()
    print("Saved: plots/precision_recall.png")

if __name__ == "__main__":
    df, X_test, y_test, predictions, model = load_data_and_model()

    plot_class_distribution(df)
    plot_confusion_matrix(y_test, predictions)
    plot_feature_importance(model, X_test)
    plot_precision_recall(y_test, predictions)

    print("\nAll plots saved to the plots/ folder.")
