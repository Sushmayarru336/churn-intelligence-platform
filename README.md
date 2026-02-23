📊 Churn Intelligence Platform

A business-focused AI-powered churn prediction and retention strategy platform built using Streamlit and machine learning.

This project allows organizations to analyze customer data, train predictive models, estimate revenue risk, and generate executive-level retention strategies using AI.

🚀 Project Overview

Customer churn is one of the biggest challenges for subscription-based and service-driven businesses.

This platform helps companies:

Predict which customers are likely to churn

Understand model performance clearly

Estimate revenue at risk

Segment customers based on churn probability

Generate AI-powered business strategy recommendations

The goal is not just prediction — but actionable intelligence.

🧠 Core Features
1️⃣ Model Training Engine

Train and evaluate multiple ML models:

Logistic Regression

Random Forest

XGBoost

Gradient Boosting

Extra Trees

Includes:

Accuracy

ROC-AUC

Recall (Churn detection strength)

Automatic churn rate calculation

2️⃣ Dashboard & Prediction

Customer-level churn probability

Predicted churn label

Revenue at risk estimation

Segment-based filtering

Clean business dashboard view

3️⃣ AI Strategy Generator

Using OpenAI API, the platform generates:

Executive-level churn interpretation

Financial impact explanation

Targeted retention strategies

Resource allocation recommendations

KPI monitoring guidance

If the API is not connected, the system safely shows a fallback message instead of crashing.

🛠 Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

XGBoost

OpenAI API

📂 Project Structure
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
    └── secrets.toml (not uploaded to GitHub)
⚙️ Local Setup Instructions
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/churn-intelligence-platform.git
cd churn-intelligence-platform
2. Create Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
🔐 Add OpenAI API Key

Create a folder:

.streamlit

Inside it create:

secrets.toml

Add:

OPENAI_API_KEY = "your_api_key_here"

⚠️ Never push this file to GitHub.

▶️ Run the Application
streamlit run Home.py

The app will open in your browser.

📈 Business Impact

This platform is useful for:

Telecom companies

SaaS businesses

Subscription platforms

Fintech companies

Insurance providers

It enables data-driven retention decisions instead of reactive churn management.

🎯 Future Improvements

Model explainability (SHAP integration)

Automated hyperparameter tuning

Customer segmentation clustering

Deployment with Docker

CI/CD pipeline integration

👤 Author

Built as an end-to-end machine learning business solution.

📜 License

MIT License