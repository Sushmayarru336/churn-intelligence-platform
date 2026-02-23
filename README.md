# 📊 Churn Intelligence Platform

An end-to-end AI-powered churn prediction and customer retention strategy platform built using Streamlit and machine learning.

This project helps businesses analyze customer data, train predictive models, evaluate performance, estimate revenue at risk, and generate executive-level retention strategies using AI.

---

## 🚀 Project Overview

Customer churn directly impacts revenue, growth, and long-term sustainability.  
This platform transforms raw customer data into actionable business intelligence.

With this application, organizations can:

- Upload customer datasets
- Train multiple machine learning models
- Evaluate model performance using business-focused metrics
- Predict churn probability at customer level
- Estimate financial risk exposure
- Generate AI-driven executive retention strategies

The objective is not just prediction — but **strategic decision-making support**.

---

## 🧠 Core Features

### 1️⃣ Model Training Engine

Supports multiple machine learning algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Extra Trees Classifier

Performance Metrics:

- Accuracy
- Recall (Churn detection strength)
- ROC-AUC Score
- Churn Rate Calculation
- Model Comparison

Session state management ensures that trained models and results remain available when navigating between pages.

---

### 2️⃣ Dashboard & Prediction Module

- Customer-level churn probability scoring
- Predicted churn classification
- Revenue at risk estimation
- Segment-based filtering (High / Medium / Low risk)
- Clean business-ready dashboard presentation

Designed to give stakeholders clear and understandable insights.

---

### 3️⃣ AI Strategy Generator

Integrated with OpenAI API to generate:

- Executive-level churn interpretation
- Financial impact breakdown
- Data-driven retention strategies
- Resource allocation recommendations
- Risk mitigation KPIs

If the API is unavailable or quota is exceeded, the system safely displays a fallback message instead of throwing an error.

---

## 🛠 Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- OpenAI API

---

## 📂 Project Structure

churn-intelligence-platform/

│
├── Home.py  
├── Strategy_and_Report.py  
├── requirements.txt  
├── README.md  
│
├── pages/  
│   ├── 1_Model_Training.py  
│   ├── 2_Dashboard_and_Prediction.py  
│
└── .streamlit/  
    └── secrets.toml  (Not pushed to GitHub)

---

## ⚙️ Local Setup Instructions

### 1️. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-intelligence-platform.git
cd churn-intelligence-platform
```

---

### 2️. Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️. Configure OpenAI API Key

Create a folder named:

```
.streamlit
```

Inside it, create a file:

```
secrets.toml
```

Add the following:

```
OPENAI_API_KEY = "your_api_key_here"
```

---

### 5️. Run the Application

```bash
streamlit run Home.py
```

The application will open in your browser.

---

## 📈 Business Applications

This solution is particularly valuable for:

- Telecom companies
- SaaS platforms
- Subscription-based businesses
- Fintech firms
- Insurance providers
- E-commerce platforms

It enables proactive retention strategies instead of reactive churn handling.

---

## 🔮 Future Enhancements

- SHAP-based model explainability
- Automated hyperparameter tuning
- Advanced customer segmentation
- Docker containerization
- CI/CD pipeline integration
- Cloud deployment scaling

---

## 👤 Author - Sushma Yarru

Developed as a complete business-oriented machine learning solution integrating predictive analytics and AI-driven strategic recommendations.

---

## 📜 License

MIT License
