import streamlit as st
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIStatusError
import os


def generate_ai_strategy(
    segment,
    segment_count,
    total_customers,
    churn_rate,
    avg_probability,
    revenue_at_risk,
    recall,
    auc
):

    segment_percentage = round((segment_count / total_customers) * 100, 2)

    # -----------------------------
    # Check if API key exists
    # -----------------------------
    if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
        return f"""
⚠️ AI Strategy Engine Not Connected

The OpenAI API key is missing or not configured properly.

Please verify:
• secrets.toml contains OPENAI_API_KEY
• Billing is active in OpenAI dashboard

Until then, below is a rule-based strategic summary:

Segment: {segment}
Customers: {segment_count} ({segment_percentage}% of base)
Avg Churn Probability: {round(avg_probability,3)}
Revenue at Risk: ${round(revenue_at_risk,2) if revenue_at_risk else 0}

Recommended Action:
Prioritize proactive engagement for this segment. 
Monitor churn drivers and deploy targeted retention campaigns 
based on behavioral indicators.
"""

    # -----------------------------
    # Try AI Generation
    # -----------------------------
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        prompt = f"""
        You are a senior enterprise churn analytics consultant.

        Business Overview:
        - Total customers analyzed: {total_customers}
        - Overall churn rate: {round(churn_rate,2)}%
        - Model Recall: {round(recall*100,2) if recall else 'N/A'}%
        - Model ROC-AUC: {round(auc,3) if auc else 'N/A'}

        Segment Analysis:
        - Segment: {segment}
        - Customers in segment: {segment_count}
        - Segment percentage: {segment_percentage}%
        - Average churn probability in segment: {round(avg_probability,3)}
        - Estimated revenue at risk: ${round(revenue_at_risk,2) if revenue_at_risk else 0}

        Provide:
        1. Strategic business interpretation (not generic)
        2. Financial impact explanation
        3. Data-driven retention actions
        4. Resource allocation recommendation
        5. Risk mitigation KPIs

        Avoid generic answers. Use the numbers above in your reasoning.
        Make it executive-level and LinkedIn-worthy.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in predictive analytics and retention strategy."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    # -----------------------------
    # If quota exceeded
    # -----------------------------
    except RateLimitError:
        return f"""
⚠️ AI Quota Exceeded

Your OpenAI usage limit has been reached.

Segment: {segment}
Customers: {segment_count} ({segment_percentage}%)
Revenue at Risk: ${round(revenue_at_risk,2) if revenue_at_risk else 0}

Recommended Interim Action:
Prioritize high-probability churn customers and deploy 
short-term retention incentives while API access is restored.
"""

    # -----------------------------
    # If internet / connection error
    # -----------------------------
    except APIConnectionError:
        return """
⚠️ AI Service Unavailable

Unable to connect to OpenAI servers.
Please check your internet connection.

Displaying fallback strategic recommendation.
"""

    # -----------------------------
    # Any other API error
    # -----------------------------
    except APIStatusError:
        return """
⚠️ AI Service Temporarily Unavailable

The AI engine encountered an issue.
Please try again later.
"""

    # -----------------------------
    # Catch all
    # -----------------------------
    except Exception:
        return """
⚠️ AI Strategy Engine Not Available

The AI module is currently not functioning.

Please verify:
• API Key
• Billing Status
• Internet Connection

Fallback strategy has been activated.
"""