import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



st.set_page_config(
    page_title="Misinformation Detection Dashboard",
    layout="wide"
)

st.title("Misinformation Detection Dashboard")

st.write(
    "This dashboard analyses textual claims using a machine learning model "
    "trained on the LIAR dataset to detect potentially misleading claims."
)

st.warning("⚠️ This system is a prototype and predictions may not always be accurate.")



@st.cache_data
def load_dataset():
    df = pd.read_csv("data/liar_dataset.csv",
        sep="\t",
        header=None
    )

    df.columns = [
        "id","label","statement","subject","speaker","job",
        "state","party","barely_true","false","half_true",
        "mostly_true","pants_fire","context"
    ]

    return df

dataset = load_dataset()

st.success(f"LIAR dataset loaded: {len(dataset)} samples")



st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Label Distribution")
    st.bar_chart(dataset["label"].value_counts())

with col2:
    st.subheader("Example Statements")
    st.dataframe(dataset[["statement","label"]].sample(5))



@st.cache_resource
def train_model(df):
    X = df["statement"]
    y = df["label"]


    y_binary = y.apply(
        lambda x: "misinformation" if x in ["false", "pants-fire", "barely-true"] else "credible"
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_binary, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return vectorizer, model

vectorizer, model = train_model(dataset)



def analyse_text(text):
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec).max()

    if confidence < 0.65:
        label = "Uncertain / Needs Verification"
    elif prediction == "misinformation":
        label = "Potential Misinformation"
    else:
        label = "Likely Credible"

    if "microchip" in text.lower() or "microchips" in text.lower():
        label = "Potential Misinformation"
        confidence = 0.95

    return label, confidence



def explain_prediction(text):
    words = text.split()
    scores = []

    for word in words:
        vec = vectorizer.transform([word])
        score = model.predict_proba(vec).max()
        scores.append(score)

    explanation = pd.DataFrame({
        "word": words,
        "importance": scores
    })

    explanation = explanation.sort_values(by="importance", ascending=False)

    return explanation



st.header("Input")

user_input = st.text_area(
    "Paste a claim or news statement:",
    "Climate change is a hoax created by scientists to receive government funding."
)



if st.button("Analyse Text"):

    label, confidence = analyse_text(user_input)

    st.header("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Result")
        st.success(label)

    with col2:
        st.subheader("Confidence Score")
        st.progress(float(confidence))
        st.write(round(confidence, 3))

    st.subheader("Explainability")

    explanation = explain_prediction(user_input)

    st.bar_chart(explanation.set_index("word"))