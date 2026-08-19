import streamlit as st
import pandas as pd

import os
from dotenv import load_dotenv


load_dotenv()
from langchain_core.tools import tool
from langchain_groq import ChatGroq
df = pd.read_csv("data/sales.csv")
@tool
def calculate_total_revenue():
    """Calculate the total revenue in the uploaded sales data."""
    return df["Revenue"].sum()

@tool
def revenue_by_city():
    """Calculate total revenue for each city."""
    return df.groupby("City")["Revenue"].sum().to_dict()

@tool
def revenue_by_product():
    """Calculate total revenue for each product."""
    return df.groupby("Product")["Revenue"].sum().to_dict()
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
tools = [
    calculate_total_revenue,
    revenue_by_city,
    revenue_by_product
]

llm_with_tools = llm.bind_tools(tools)

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




st.header("🤖 Ask Data Genie")
question = st.chat_input(
    "💬 Ask a question about your data...",
    key="data_genie_chat"
)
if question:
    response = llm_with_tools.invoke(question)

    if response.tool_calls:
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            if tool_name == "calculate_total_revenue":
                result = calculate_total_revenue.invoke({})

            elif tool_name == "revenue_by_city":
                result = revenue_by_city.invoke({})

            elif tool_name == "revenue_by_product":
                result = revenue_by_product.invoke({})

            final_prompt = f"""
            You are a Business Intelligence assistant.

            The user asked:
            {question}

            The data tool returned:
            {result}

            Give the user a clear and simple business answer.
            """

            final_response = llm.invoke(final_prompt)

            st.write(final_response.content)

    else:
        st.write(response.content)
