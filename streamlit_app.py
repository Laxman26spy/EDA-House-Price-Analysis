
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price EDA",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ames Housing Price Analysis")
st.write("Exploratory Data Analysis of House Prices")

file = st.file_uploader("Upload train.csv", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.success("Dataset loaded successfully!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Houses", len(df))
    c2.metric("Total Features", len(df.columns))
    c3.metric("Average Price", f"${df['SalePrice'].mean():,.0f}")

    st.subheader("1. House Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["SalePrice"], kde=True, ax=ax)
    ax.set_xlabel("Sale Price")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2. Overall Quality vs Sale Price")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="OverallQual",
                y="SalePrice", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Living Area vs Sale Price")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="GrLivArea",
                    y="SalePrice", hue="OverallQual",
                    ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Correlation Heatmap")
    numeric = df.select_dtypes(include="number")
    top = numeric.corr()["SalePrice"].abs().nlargest(10).index

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric[top].corr(),
                annot=True, cmap="coolwarm",
                fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Dataset Preview")
    st.dataframe(df.head(10))

else:
    st.info("Please upload the Ames Housing train.csv file to begin.")
