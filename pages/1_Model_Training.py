import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

st.set_page_config(page_title="Model Training", layout="wide")
st.title("📊 Advanced Churn Model Training Engine")

# -----------------------------
# Initialize session state
# -----------------------------
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "download_done" not in st.session_state:
    st.session_state.download_done = False

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Logistic Regression"

if "target_column" not in st.session_state:
    st.session_state.target_column = None

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])

# If new file uploaded → reset session and store new df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_df = df
    st.session_state.model_ready = False
    st.session_state.download_done = False
    st.session_state.target_column = None
    st.session_state.model_choice = "Logistic Regression"

# -----------------------------
# If data exists in session → show everything
# -----------------------------
if st.session_state.uploaded_df is not None:

    df = st.session_state.uploaded_df

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # TARGET COLUMN (PERSIST)
    # -----------------------------
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        default_target_index = list(df.columns).index(st.session_state.target_column)
    else:
        default_target_index = 0

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        index=default_target_index
    )

    st.session_state.target_column = target_column

    if df[target_column].dtype == "object":
        df[target_column] = df[target_column].map({"No": 0, "Yes": 1})

    if df[target_column].dropna().nunique() == 2:
        churn_rate = df[target_column].mean() * 100
        st.metric("Overall Churn Rate", f"{round(churn_rate,2)}%")
        st.session_state["churn_rate"] = churn_rate
    else:
        st.warning("Selected column which can be able to predict churn")
        st.stop()

    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(int)

    # -----------------------------
    # MODEL SELECTION (PERSIST)
    # -----------------------------
    model_list = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Gradient Boosting",
        "Extra Trees",
    ]

    default_model_index = model_list.index(st.session_state.model_choice)

    model_choice = st.selectbox(
        "Select Model",
        model_list,
        index=default_model_index
    )

    st.session_state.model_choice = model_choice

    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # -----------------------------
    # TRAIN MODEL BUTTON
    # -----------------------------
    if st.button("🚀 Train Model"):

        with st.spinner("Model is training... Please wait a few seconds ⏳"):
            time.sleep(2)

            if model_choice == "Logistic Regression":
                classifier = LogisticRegression(max_iter=1000)

            elif model_choice == "Random Forest":
                classifier = RandomForestClassifier(n_estimators=200, random_state=42)

            elif model_choice == "XGBoost":
                classifier = XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric="logloss"
                )

            elif model_choice == "Gradient Boosting":
                classifier = GradientBoostingClassifier()

            elif model_choice == "Extra Trees":
                classifier = ExtraTreesClassifier(n_estimators=200, random_state=42)

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model.fit(X_train, y_train)

            full_prob = model.predict_proba(X)[:, 1]
            full_pred = model.predict(X)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            recall = recall_score(y_test, y_pred)

            df_download = df.copy()
            df_download["Predicted_Churn"] = full_pred
            df_download["Churn_Probability"] = full_prob

            # Store everything in session
            st.session_state["model"] = model
            st.session_state["model_ready"] = True
            st.session_state["accuracy"] = accuracy
            st.session_state["auc"] = auc
            st.session_state["recall"] = recall
            st.session_state["trained_df"] = df_download

        st.success(f"✅ {model_choice} trained successfully!")

    # -----------------------------
    # SHOW METRICS AFTER TRAINING
    # -----------------------------
    if st.session_state.model_ready:

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(st.session_state.accuracy,3))
        col2.metric("ROC-AUC", round(st.session_state.auc,3))
        col3.metric("Recall (Churn Detection)", round(st.session_state.recall,3))

        download_clicked = st.download_button(
            label="⬇ Download Trained Dataset with Predictions",
            data=st.session_state.trained_df.to_csv(index=False),
            file_name="trained_dataset_with_predictions.csv",
            mime="text/csv"
        )

        if download_clicked:
            st.session_state.download_done = True

        if st.session_state.download_done:
            st.success("📥 You have successfully downloaded the trained dataset!")
            st.info("You can now navigate to Dashboard & Prediction page from the sidebar.")