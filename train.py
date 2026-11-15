import pandas as pd
from pathlib import Path

# Find the location of this Python file
base = Path(__file__).resolve().parent

# Search for Fake.csv and True.csv
def find_file(filename):
    locations = [
        base / filename,
        base / "dataset" / filename,
        base.parent / filename,
        base.parent / "dataset" / filename
    ]

    for path in locations:
        if path.exists():
            return path

    raise FileNotFoundError(f"{filename} not found. Check its location.")

# Load datasets
fake_path = find_file("Fake.csv")
true_path = find_file("True.csv")

fake = pd.read_csv(fake_path)
real = pd.read_csv(true_path)

# Display data
print("FAKE NEWS DATA")
print(fake.head())

print("\nREAL NEWS DATA")
print(real.head())

print("\nFake News Shape:", fake.shape)
print("Real News Shape:", real.shape)

print("\nDataset loaded successfully!")
# Assign labels
fake["label"] = 0
real["label"] = 1

# Combine both datasets
data = pd.concat([fake, real], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display combined dataset
print("\nCOMBINED NEWS DATA")
print(data.head())

# Display total records
print("\nTotal Dataset Shape:", data.shape)

# Save combined dataset
data.to_csv(base / "news_dataset.csv", index=False)

print("\nCombined dataset saved successfully!")


import re

# Combine title and news text
data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Apply cleaning
data["clean_text"] = data["content"].apply(clean_text)

# Display cleaned data
print("\nCLEANED NEWS DATA")
print(data[["clean_text", "label"]].head())

# Save cleaned dataset
data.to_csv(base / "news_dataset_cleaned.csv", index=False)

print("\nText preprocessing completed!")
print("Cleaned dataset saved successfully!")
# STEP 5: CONVERT TEXT INTO NUMBERS

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("\nTF-IDF CONVERSION COMPLETED!")
print("Feature Matrix Shape:", X.shape)
print("Labels Shape:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDATASET SPLIT COMPLETED!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict results
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL TESTING COMPLETED!")
print("Model Accuracy:", accuracy * 100, "%")

# Display performance report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, base / "fake_news_model.pkl")
joblib.dump(vectorizer, base / "tfidf_vectorizer.pkl")

print("\nMODEL AND VECTORIZER SAVED SUCCESSFULLY!")


