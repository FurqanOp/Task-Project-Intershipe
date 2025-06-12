import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load all JSON files into a single list
# ----------------------------

json_paths = [
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_agenda_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_agreement_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_annex_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_announcement_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_appendix_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_audit_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_brochure_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_business_case_documents.json",
    r"C:\Users\furqan ahmed\Desktop\Task & Project Intershipe\json\files\synthetic_documents.json"
]

documents = []

for file_path in json_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            content = doc.get("content", "")
            label = doc.get("label", "")
            taxonomy = doc.get("taxonomy") or doc.get("category") or doc.get("title")
            if taxonomy:
                taxonomy = [taxonomy] if isinstance(taxonomy, str) else taxonomy
            else:
                taxonomy = []

            documents.append({
                "text": content,
                "label": label,
                "taxonomy": taxonomy
            })

df = pd.DataFrame(documents)

# ----------------------------
# 2. Filter rare taxonomy labels (<2 occurrences)
# ----------------------------

flat_labels = [label for sublist in df["taxonomy"] for label in sublist]
label_counts = Counter(flat_labels)

def keep_valid_labels(row):
    return all(label_counts[label] >= 2 for label in row["taxonomy"])

df = df[df.apply(keep_valid_labels, axis=1)].reset_index(drop=True)

# ----------------------------
# 3. (Optional) Plot label distribution
# ----------------------------
label_series = pd.Series([l for lst in df["taxonomy"] for l in lst])
plt.figure(figsize=(10, 5))
label_series.value_counts().plot(kind="bar")
plt.title("📌 Taxonomy Label Distribution (Filtered)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Document Type Classifier (Single Label)
# ----------------------------

X = df["text"]
y = df["label"]

X_train_doc, X_test_doc, y_train_doc, y_test_doc = train_test_split(X, y, test_size=0.3, random_state=42)

doc_type_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

doc_type_clf.fit(X_train_doc, y_train_doc)
y_pred_doc = doc_type_clf.predict(X_test_doc)

print("\n=== 📄 Document Type Classification Report ===")
print(classification_report(y_test_doc, y_pred_doc, zero_division=0))
print("Accuracy:", accuracy_score(y_test_doc, y_pred_doc))

# ----------------------------
# 5. Taxonomy Classifier (Multi-Label) with GridSearchCV
# ----------------------------

mlb = MultiLabelBinarizer()
Y_multi = mlb.fit_transform(df["taxonomy"])

X_train_tax, X_test_tax, y_train_tax, y_test_tax = train_test_split(X, Y_multi, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))
])

param_grid = {
    'tfidf__max_df': [0.8, 1.0],
    'tfidf__min_df': [1, 3],
    'clf__estimator__C': [0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1_micro',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_tax, y_train_tax)
y_pred_tax = grid.predict(X_test_tax)

print("\n=== ✅ Best Parameters from GridSearch ===")
print(grid.best_params_)

print("\n=== 🧾 Taxonomy Classification Report (Optimized) ===")
print(classification_report(y_test_tax, y_pred_tax, target_names=mlb.classes_, zero_division=0))

print("\n=== 🎯 Accuracy per Taxonomy Label ===")
for idx, label in enumerate(mlb.classes_):
    acc = accuracy_score(y_test_tax[:, idx], y_pred_tax[:, idx])
    print(f"{label:25}: {acc:.2f}")
