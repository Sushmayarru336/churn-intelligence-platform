import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime
from Strategy_and_Report import generate_ai_strategy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


st.title("📈 Dashboard & Prediction")


# -----------------------------
# Safe Session Initialization
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "trained_df" not in st.session_state:
    st.session_state.trained_df = None

if "churn_rate" not in st.session_state:
    st.session_state.churn_rate = None



if not st.session_state.model_ready:
    st.warning("⚠️ Please train a model first from Training page.")
    st.stop()

model = st.session_state.model
trained_df = st.session_state.trained_df
churn_rate = st.session_state.churn_rate
recall = st.session_state.recall
auc = st.session_state.auc
target_column = st.session_state["target_column"]

# -------------------------------------------------
# TRAINING METRICS
# -------------------------------------------------
st.subheader("Model Performance (Training)")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{round(st.session_state['accuracy']*100,2)}%")
col2.metric("ROC-AUC", round(st.session_state["auc"],3))

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Dataset for Prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    HIGH_RISK_THRESHOLD = st.slider("High Risk Threshold", 0.5, 0.95, 0.75)
    MEDIUM_RISK_THRESHOLD = 0.40

    if st.button("Predict Churn"):

        if target_column in df.columns:
            actual = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            actual = None
            X = df

        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

        df["Churn_Probability"] = probabilities
        df["Predicted_Churn"] = predictions

        # Segmentation
        def segment_customer(prob):
            if prob >= HIGH_RISK_THRESHOLD:
                return "High Risk"
            elif prob >= MEDIUM_RISK_THRESHOLD:
                return "Medium Risk"
            else:
                return "Low Risk"

        df["Segment"] = df["Churn_Probability"].apply(segment_customer)

        # Risk counts
        high_risk_count = len(df[df["Segment"] == "High Risk"])
        medium_risk_count = len(df[df["Segment"] == "Medium Risk"])
        low_risk_count = len(df[df["Segment"] == "Low Risk"])
        
        # Revenue
        if "MonthlyCharges" in df.columns:
            revenue_at_risk = df[df["Segment"] == "High Risk"]["MonthlyCharges"].sum()
        else:
            revenue_at_risk = None

        # Validation
        if actual is not None:
            accuracy = accuracy_score(actual, predictions)
            precision = precision_score(actual, predictions)
            recall = recall_score(actual, predictions)
            f1 = f1_score(actual, predictions)
            auc = roc_auc_score(actual, probabilities)
            cm = confusion_matrix(actual, predictions)
            report = classification_report(actual, predictions)
        else:
            accuracy = precision = recall = f1 = auc = None
            cm = report = None

        # Save to session
        st.session_state["prediction_df"] = df
        st.session_state["metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "cm": cm,
            "report": report
        }
        st.session_state["high"] = high_risk_count
        st.session_state["medium"] = medium_risk_count
        st.session_state["low"] = low_risk_count
        st.session_state["revenue"] = revenue_at_risk
        st.session_state["churn_rate"] = st.session_state.get("accuracy", 0) * 100

# =================================================
# DISPLAY RESULTS
# =================================================
if "prediction_df" in st.session_state:

    df = st.session_state["prediction_df"]
    metrics = st.session_state["metrics"]

    high = st.session_state["high"]
    medium = st.session_state["medium"]
    low = st.session_state["low"]
    revenue = st.session_state["revenue"]

    # -------------------------------------------------
    # PIE CHART
    # -------------------------------------------------
    st.subheader("Customer Risk Distribution")

    fig, ax = plt.subplots()
    ax.pie(
        [high, medium, low],
        labels=["High Risk", "Medium Risk", "Low Risk"],
        autopct="%1.1f%%"
    )
    ax.set_title("Risk Distribution")
    st.pyplot(fig)

    # -------------------------------------------------
    # DROPDOWN FILTER
    # -------------------------------------------------
    st.subheader("Filter Customers by Segment")

    selected_segment = st.selectbox(
        "Select Risk Category",
        ["High Risk", "Medium Risk", "Low Risk"]
    )

    filtered_df = df[df["Segment"] == selected_segment]

    col1, col2 = st.columns([3, 1])
    col1.dataframe(filtered_df)
    col2.metric(f"{selected_segment} Count", len(filtered_df))

    st.download_button(
        f"Download {selected_segment} Customers",
        filtered_df.to_csv(index=False),
        file_name=f"{selected_segment}_customers.csv",
        mime="text/csv"
    )

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    if metrics["accuracy"] is not None:

        st.subheader("Model Validation")

        st.metric("Accuracy", f"{round(metrics['accuracy']*100,2)}%")
        st.metric("Recall", f"{round(metrics['recall']*100,2)}%")

        st.text(metrics["report"])
        st.write("Confusion Matrix")
        st.write(metrics["cm"])
    
        # -------------------------------------------------
        # AI STRATEGY GENERATOR
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🧠 AI Retention Strategy")

        total_customers = high + medium + low
        churn_rate = st.session_state.get("churn_rate", 0)

        segment_option = st.selectbox(
            "Select Segment for AI Strategy",
            ["High Risk", "Medium Risk", "Low Risk"]
        )

        segment_count = {
            "High Risk": high,
            "Medium Risk": medium,
            "Low Risk": low
        }[segment_option]

        if st.button("Generate AI Strategy"):

            with st.spinner("Generating Intelligent Strategy..."):

                avg_probability = filtered_df["Churn_Probability"].mean()

                strategy_text = generate_ai_strategy(
                    segment=segment_option,
                    segment_count=segment_count,
                    total_customers=total_customers,
                    churn_rate=churn_rate,
                    avg_probability=avg_probability,
                    revenue_at_risk=revenue if revenue else 0,
                    recall=metrics["recall"],
                    auc=metrics["auc"]
                )

                st.session_state["ai_strategy"] = strategy_text
                st.success("Advanced AI Strategy Generated!")

        if "ai_strategy" in st.session_state:

            st.markdown("### 📌 Strategy Output")
            st.write(st.session_state["ai_strategy"])

            # Export AI Report
            report_text = f"""
            CHURN STRATEGY REPORT
            =====================

            Overall Churn Rate: {round(churn_rate,2)}%

            High Risk Customers: {high}
            Medium Risk Customers: {medium}
            Low Risk Customers: {low}

            Selected Segment: {segment_option}

            AI STRATEGY:
            ------------
            {st.session_state['ai_strategy']}
            """

            st.download_button(
                label="📥 Download AI Strategy Report",
                data=report_text,
                file_name="AI_Churn_Strategy_Report.txt",
                mime="text/plain"
            )

    # -------------------------------------------------
    # PDF GENERATION
    # -------------------------------------------------
    if st.button("Generate overall Executive PDF Report"):

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Logo
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            elements.append(Image(logo_path, width=120, height=60))
            elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Enterprise Churn Prediction Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Generated on: {datetime.now()}", styles["Normal"]))
        elements.append(Spacer(1, 0.5 * inch))

        # Risk Summary
        elements.append(Paragraph("Risk Summary", styles["Heading2"]))
        elements.append(Paragraph(f"High Risk: {high}", styles["Normal"]))
        elements.append(Paragraph(f"Medium Risk: {medium}", styles["Normal"]))
        elements.append(Paragraph(f"Low Risk: {low}", styles["Normal"]))

        if revenue is not None:
            elements.append(Paragraph(
                f"Estimated Revenue at Risk: ${round(revenue,2)}",
                styles["Normal"]
            ))

        elements.append(Spacer(1, 0.4 * inch))

        # Add Pie Chart to PDF
        pie_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(pie_path)
        elements.append(Image(pie_path, width=400, height=300))

        elements.append(Spacer(1, 0.4 * inch))

        # Validation
        if metrics["accuracy"] is not None:

            elements.append(Paragraph("Model Performance", styles["Heading2"]))
            elements.append(Paragraph(f"Accuracy: {round(metrics['accuracy']*100,2)}%", styles["Normal"]))
            elements.append(Paragraph(f"Precision: {round(metrics['precision']*100,2)}%", styles["Normal"]))
            elements.append(Paragraph(f"Recall: {round(metrics['recall']*100,2)}%", styles["Normal"]))
            elements.append(Paragraph(f"F1 Score: {round(metrics['f1']*100,2)}%", styles["Normal"]))
            elements.append(Paragraph(f"ROC-AUC: {round(metrics['auc'],3)}", styles["Normal"]))

            elements.append(Spacer(1, 0.3 * inch))

            # Confusion Matrix Image
            fig_cm, ax_cm = plt.subplots()
            ax_cm.matshow(metrics["cm"])
            plt.title("Confusion Matrix")
            cm_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig_cm.savefig(cm_path)
            plt.close(fig_cm)

            elements.append(Image(cm_path, width=300, height=300))

        # Executive Summary
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("Executive Summary", styles["Heading2"]))

        if metrics["recall"] is not None:
            if metrics["recall"] > 0.80:
                summary = "The model strongly identifies churners and enables proactive retention."
            elif metrics["recall"] > 0.60:
                summary = "The model provides moderate churn detection capability."
            else:
                summary = "Model recall suggests further optimization may improve churn detection."
        else:
            summary = "Validation metrics unavailable."

        elements.append(Paragraph(summary, styles["Normal"]))

        doc.build(elements)

        with open(temp_pdf.name, "rb") as f:
            st.download_button(
                "Download Executive PDF",
                f,
                file_name="Churn_Executive_Report.pdf",
                mime="application/pdf"
            )