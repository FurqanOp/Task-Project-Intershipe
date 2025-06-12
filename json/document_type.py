import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 🔧 Set page config at the very top
st.set_page_config(page_title="Document Classifier", layout="wide")

# -------------------------------
# Load Data and Train Models
# -------------------------------
@st.cache_resource
def train_models():
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
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            for doc in docs:
                content = doc.get("content", "")
                label = doc.get("label", "")
                taxonomy = doc.get("taxonomy") or doc.get("category") or doc.get("title")
                if taxonomy:
                    taxonomy = [taxonomy] if isinstance(taxonomy, str) else taxonomy
                else:
                    taxonomy = []
                documents.append({"text": content, "label": label, "taxonomy": taxonomy})

    df = pd.DataFrame(documents)

    # Filter rare taxonomy labels
    from collections import Counter
    flat_labels = [label for sublist in df["taxonomy"] for label in sublist]
    label_counts = Counter(flat_labels)

    def keep_valid_labels(row):
        return all(label_counts[label] >= 2 for label in row["taxonomy"])

    df = df[df.apply(keep_valid_labels, axis=1)].reset_index(drop=True)

    # Train classifiers
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    doc_type_clf = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    doc_type_clf.fit(X_train, y_train)
    y_pred = doc_type_clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=doc_type_clf.classes_)

    # Multi-label taxonomy
    mlb = MultiLabelBinarizer()
    Y_multi = mlb.fit_transform(df["taxonomy"])

    tax_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))
    ])
    tax_model.fit(X, Y_multi)

    return doc_type_clf, tax_model, mlb, report, conf_mat, doc_type_clf.classes_

# Load models
doc_model, tax_model, label_binarizer, clf_report, conf_matrix, doc_classes = train_models()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄 Document Classifier App")
st.markdown("Upload or paste a document and get its predicted **type** and **taxonomy labels**.")

user_input = st.text_area("✍️ Paste your document content here:", height=250)

if st.button("🔍 Predict"):
    if user_input.strip():
        # Predict document type
        doc_type_pred = doc_model.predict([user_input])[0]

        # Predict taxonomy using top 5 probabilities
        probas = tax_model.predict_proba([user_input])[0]
        top_indices = probas.argsort()[-5:][::-1]
        top_labels = [label_binarizer.classes_[i] for i in top_indices if probas[i] > 0.1]
        taxonomy_labels = top_labels[:5]

        st.success("✅ Prediction Complete!")
        st.markdown(f"**📄 Document Type:** `{doc_type_pred}`")
        st.markdown("**🏷️ Taxonomy Labels:**")
        st.write(taxonomy_labels if taxonomy_labels else "No matching taxonomy found.")
    else:
        st.warning("⚠️ Please enter some document content.")

# -------------------------------
# Evaluation Metrics Section
# -------------------------------
if st.checkbox("📊 Show evaluation metrics"):
    st.subheader("📄 Document Type Classification Report")
    st.dataframe(pd.DataFrame(clf_report).transpose().round(2))

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=doc_classes, yticklabels=doc_classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
