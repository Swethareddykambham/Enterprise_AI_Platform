import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Business Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown(
    """
<style>

.main{
    background:#F6F8FC;
}

.block-container{
    padding-top:2rem;
}

.big-title{
    font-size:42px;
    font-weight:700;
    color:#1F2937;
}

.subtitle{
    font-size:18px;
    color:#6B7280;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:18px;
    box-shadow:0 5px 20px rgba(0,0,0,.08);
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown(
    "<div class='big-title'>🤖 Enterprise AI Business Intelligence Platform</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>Retail Analytics | NLP | Deep Learning | Forecasting | Business Intelligence</div>",
    unsafe_allow_html=True,
)

st.divider()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🧠 Customer Analytics",
        "💬 NLP",
        "📈 Sequence Models",
        "👥 Customer Segmentation",
        "⚙ Feature Engineering",
        "📊 Time Series",
        "📑 Reports",
        "ℹ About",
    ],
)

# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------

if page == "🏠 Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        "25,450",
        "+12%"
    )

    col2.metric(
        "Revenue",
        "$2.48M",
        "+8%"
    )

    col3.metric(
        "Orders",
        "15,920",
        "+16%"
    )

    col4.metric(
        "AI Accuracy",
        "96.8%",
        "+2%"
    )

    st.divider()

    sales = pd.DataFrame(
        {
            "Month": [
                "Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"
            ],
            "Revenue": [
                120,150,180,210,240,280,
                310,350,390,420,460,520
            ]
        }
    )

    fig = px.line(
        sales,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    category = pd.DataFrame(
        {
            "Category":[
                "Electronics",
                "Fashion",
                "Furniture",
                "Food"
            ],
            "Sales":[42,28,15,15]
        }
    )

    fig2 = px.pie(
        category,
        values="Sales",
        names="Category",
        hole=0.45,
        title="Sales Distribution"
    )

    left.plotly_chart(
        fig2,
        use_container_width=True
    )

    customers = pd.DataFrame(
        {
            "Age":np.random.randint(
                18,
                60,
                300
            ),
            "Spending":np.random.randint(
                100,
                1500,
                300
            )
        }
    )

    fig3 = px.scatter(
        customers,
        x="Age",
        y="Spending",
        color="Spending",
        title="Customer Spending Pattern"
    )

    right.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader("Recent Business Insights")

    st.success(
        "Revenue increased by 8% compared to the previous month."
    )

    st.info(
        "Customer retention has improved after launching the loyalty program."
    )

    st.warning(
        "Inventory levels are low for Electronics."
    )