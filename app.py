import streamlit as st
import pandas as pd

st.title("       Data Genie 🤖📊     ")
st.header("   Business Intelligence Dashboard      ")
st.header("😄welcome")


uploaded_file = st.file_uploader(
    "📂 Upload your sales CSV",
    type=["csv"]
)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    st.dataframe(df)

df = pd.read_csv("data/sales.csv")

st.dataframe(df)

total_revenue = df["Revenue"].sum()
total_quantity = df["Quantity"].sum()
total_cities = df["City"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Revenue", f"Rs. {total_revenue:,}")
col2.metric("📦 Units Sold", total_quantity)
col3.metric("🏙️ Cities", total_cities)


st.header("🏙️ City Analysis")

city_sales = df.groupby("City")["Revenue"].sum()

st.bar_chart(city_sales)
st.header("📦 Product Analysis")

product_sales = df.groupby("Product")["Revenue"].sum()

st.bar_chart(product_sales)

st.header("🔎 Explore Data")

selected_city = st.selectbox(
    "Select a city",
    df["City"].unique()
)
filtered_df = df[df["City"] == selected_city]

st.dataframe(filtered_df)


best_product = df.groupby("Product")["Revenue"].sum().idxmax()

st.success(f"🏆 Best Performing Product: {best_product}")


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
st.header("🤖 Ask Data Genie")


question = st.chat_input(
    "💬 Ask a question about your data...",
    key="Data Genie"
)

if question:
    data_summary = df.to_string()

    prompt = f"""
You are a Business Intelligence assistant.

Here is the business data:

{data_summary}

Answer the user's question using the data above.

User question:
{question}
"""

    response = llm.invoke(prompt)

    st.write(response.content)

def calculate_total_revenue(df):
    return df["Revenue"].sum()
