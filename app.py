
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config(page_title="ModelSimply", layout="wide")
st.title("🧠 ModelSimply - AutoML Made Easy")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🛠 Model Configuration")
    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        target = st.selectbox("🎯 Select target column", df.columns)
        features = st.multiselect("🧠 Select feature columns", [col for col in df.columns if col != target])
        test_size = st.slider("🧪 Test size (%)", 10, 50, 20)
        model_choice = st.selectbox("🤖 Choose Model", ["Logistic Regression", "Random Forest"])

        if model_choice == "Random Forest":
            n_estimators = st.slider("🌲 Number of Trees", 10, 200, 100)

        if st.button("🚀 Train Model"):
            with col2:
                st.subheader("📊 Output & Results")

            X = df[features].copy()
            y = df[target].copy()

            # Encode non-numeric columns
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == "object":
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le

            if y.dtype == "object":
                y_le = LabelEncoder()
                y = y_le.fit_transform(y)
                label_encoders[target] = y_le

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            else:
                model = RandomForestClassifier(n_estimators=n_estimators)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            with col2:
                st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")
                st.write("### 🧾 Generated Code:")
                st.code(f"""
# Load dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.{ 'ensemble' if model_choice == 'Random Forest' else 'linear_model' } import { model.__class__.__name__ }
from sklearn.preprocessing import LabelEncoder

# Data loading and preprocessing
X = df[{features}]
y = df["{target}"]

# Encode categorical data
# (Same logic as UI)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size/100}, random_state=42)

# Model training
model = {model.__class__.__name__}({f'n_estimators={n_estimators}' if model_choice == 'Random Forest' else ''})
model.fit(X_train, y_train)
                """)

                joblib.dump(model, f"{model.__class__.__name__}.pkl")
                st.download_button("💾 Download Model", open(f"{model.__class__.__name__}.pkl", "rb").read(), file_name=f"{model.__class__.__name__}.pkl")

            with col2:
                st.write("### 📂 Data Preview:")
                st.dataframe(df.head())
