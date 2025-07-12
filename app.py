import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="AutoML GUI", layout="wide")
st.title("🤖 AutoML GUI - Build ML Models Without Code")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview:", df.head())

    target = st.selectbox("🎯 Select target column", df.columns)
    features = st.multiselect("🧠 Select feature columns", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]
        test_size = st.slider("Test size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        else:
            n_estimators = st.slider("Number of Trees (Random Forest)", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators)

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")
            st.code(f"""# Generated Code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.{ 'ensemble' if model_choice == 'Random Forest' else 'linear_model' } import { model.__class__.__name__ }

df = pd.read_csv("your_data.csv")
X = df[{features}]
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size/100}, random_state=42)
model = {model.__class__.__name__}({f'n_estimators={n_estimators}' if model_choice == 'Random Forest' else ''})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
""")

            joblib.dump(model, f"{model.__class__.__name__}.pkl")
            st.download_button("💾 Download Trained Model", data=open(f"{model.__class__.__name__}.pkl", "rb").read(), file_name=f"{model.__class__.__name__}.pkl")
