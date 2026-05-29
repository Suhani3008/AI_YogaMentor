import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("landmarks.csv")

print("Dataset Shape:")
print(df.shape)

# -----------------------------------
# Features and Labels
# -----------------------------------

X = df.drop("label", axis=1)

y = df["label"]

# -----------------------------------
# Label Encoding
# -----------------------------------

le = LabelEncoder()

y = le.fit_transform(y)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------------
# Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# -----------------------------------
# Train Model
# -----------------------------------

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed")

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Accuracy
# -----------------------------------

train_acc = model.score(X_train, y_train)

test_acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Training Accuracy : {train_acc*100:.2f}%")
print(f"Testing Accuracy  : {test_acc*100:.2f}%")
print("==============================")

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    )
)

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

# -----------------------------------
# Accuracy Bar Graph
# -----------------------------------

plt.figure(figsize=(5,5))

plt.bar(
    ["Train Accuracy", "Test Accuracy"],
    [train_acc*100, test_acc*100]
)

plt.ylabel("Accuracy (%)")

plt.title("Model Accuracy")

plt.ylim(0,100)

plt.savefig("accuracy_graph.png")

plt.show()

# -----------------------------------
# Save Model
# -----------------------------------

pickle.dump(
    model,
    open("model.pkl", "wb")
)

# -----------------------------------
# Save Label Encoder
# -----------------------------------

pickle.dump(
    le,
    open("label_encoder.pkl", "wb")
)

# -----------------------------------
# Save Scaler
# -----------------------------------

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

print("\n================================")
print("Model Saved Successfully")
print("================================")

