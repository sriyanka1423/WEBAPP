import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
import pickle
import urllib
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ------------------ DATABASE SETUP ------------------
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=retailserver2.database.windows.net;"
    "DATABASE=RetailDB;"
    "UID=sriyanka;"
    "PWD=Pubsi@1423;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
conn = engine.connect()

# ------------------ SESSION STATE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# ------------------ UI TITLE ------------------
st.set_page_config(page_title="Retail Dashboard", layout="wide")
st.title("🧠 Retail Intelligence App")

# ------------------ MENU OPTIONS ------------------
if st.session_state.logged_in:
    st.sidebar.markdown(f"👤 Welcome, *{st.session_state.username}*")

if st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Dashboard", "Upload Data", "Data Pull", "Basket ML","Churn Prediction", "Logout"])
else:
    menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

# ------------------ LOGIN ------------------
if menu == "Login":
    st.subheader("🔐 Login")
    user_input = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        query = text("""
            SELECT * FROM users 
            WHERE (LOWER(email) = :input OR LOWER(username) = :input) 
            AND password = :password
        """)
        result = conn.execute(query, {"input": user_input.lower(), "password": password}).fetchone()
        if result:
            st.session_state.logged_in = True
            st.session_state.user_email = result.email
            st.session_state.username = result.username
            st.rerun()
        else:
            st.error("Invalid username/email or password.")


# ------------------ SIGNUP ------------------
elif menu == "Signup":
    st.subheader("📝 Create Account")

    username = st.text_input("Username", key="signup_username")
    email = st.text_input("New Email", key="signup_email")
    password = st.text_input("Create Password", type="password", key="signup_password")
    confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

    if st.button("Signup", key="signup_button"):
        if not username or not email or not password or not confirm:
            st.warning("Please fill in all fields.")
        elif password != confirm:
            st.warning("Passwords do not match.")
        else:
            # Check if user already exists
            existing_user = conn.execute(
                text("SELECT * FROM users WHERE email = :email OR username = :username"),
                {"email": email, "username": username}
            ).fetchone()

            if existing_user:
                st.error("User with this email or username already exists.")
            else:
                with engine.begin() as con:
                    con.execute(
                        text("INSERT INTO users (username, email, password) VALUES (:username, :email, :password)"),
                        {"username": username, "email": email, "password": password}
                    )
                st.success("✅ Account created! Please login.")


# ------------------ LOGOUT ------------------
elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.success("🔒 Logged out successfully.")
    st.rerun()

# ------------------ DASHBOARD ------------------
elif menu == "Dashboard" and st.session_state.logged_in:
    st.subheader("📊 Creative Retail Dashboard")
    households = pd.read_sql("SELECT * FROM Households", conn)
    transactions = pd.read_sql("SELECT * FROM Transactions", conn)
    products = pd.read_sql("SELECT * FROM Products", conn)

    households.columns = households.columns.str.lower()
    transactions.columns = transactions.columns.str.lower()
    products.columns = products.columns.str.lower()

    households["loyalty_flag"] = households["l"].map({"Y": 1, "N": 0}).fillna(0)
    merged = transactions.merge(products, on="product_num").merge(households, on="hshd_num")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👨‍👩‍👧 Demographics", "📈 Time Trends", "🧺 Basket Analysis", "🕐 Seasonal Trends", "🏷️ Brand Preference"])

    with tab1:
        st.subheader("👨‍👩‍👧‍👦 Engagement by Demographics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("*Avg Spend by Household Size (Bar)*")
            fig1 = px.bar(merged.groupby("hh_size")["spend"].mean().reset_index(), x="hh_size", y="spend")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("*Avg Spend by Children (Pie)*")
            children_df = merged.groupby("children")["spend"].mean().reset_index()
            fig2 = px.pie(children_df, names="children", values="spend")
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("*Avg Spend by Income Range (Line)*")
            income_df = merged.groupby("income_range")["spend"].mean().reset_index()
            fig3 = px.line(income_df, x="income_range", y="spend")
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("📆 Engagement Over Time")
        merged["purchase_"] = pd.to_datetime(merged["purchase_"], errors='coerce')
        merged["year_month"] = merged["purchase_"].dt.to_period("M")
        trend = merged.groupby("year_month")["spend"].sum().reset_index()
        trend["year_month"] = trend["year_month"].astype(str)
        fig = px.line(trend, x="year_month", y="spend", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🛒 Product Pairings and Cross-Sell")
        combo = merged.groupby(["basket_num", "commodity"])["product_num"].count().reset_index()
        basket = combo.groupby("commodity")["product_num"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(basket, orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🕐 Seasonal Spending")
        merged["month"] = merged["purchase_"].dt.month
        season = merged.groupby("month")["spend"].sum().reset_index()
        fig = px.area(season, x="month", y="spend")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("🏷️ Brand and Organic Preferences")
        brand_col = [col for col in products.columns if "brand" in col][0]
        organic_col = [col for col in products.columns if "organic" in col][0]

        fig = px.pie(products, names=brand_col, title="Brand Preference")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(products.groupby(organic_col).size().reset_index(name="count"), x=organic_col, y="count")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------ FILE UPLOAD ------------------

elif menu == "Upload Data" and st.session_state.logged_in:
    st.subheader("📤 Upload CSV Files")

    uploaded = {}
    for label in ["Households", "Transactions", "Products"]:
        uploaded[label] = st.file_uploader(f"Upload {label}.csv", type="csv")

    if st.button("Update Azure Tables"):
        if all(uploaded.values()):
            with st.spinner('Uploading to Azure SQL Database...'):
                try:
                    # 🧠 Always re-create fresh small connection
                    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
                    conn = engine.connect()

                    for table, file in uploaded.items():
                        df = pd.read_csv(file)
                        df.columns = df.columns.str.strip()  # clean headers
                        df.to_sql(table, con=conn, if_exists="replace", index=False)

                    st.success("✅ All tables uploaded successfully to Azure!")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
        else:
            st.warning("⚠️ Please upload all three files before updating.")


# ------------------ INTERACTIVE DATA PULL ------------------
elif menu == "Data Pull" and st.session_state.logged_in:
    st.subheader("🔎 Interactive Data Pull by HSHD_NUM")
    hshd = st.number_input("Enter HSHD_NUM", value=10)
    if st.button("Get Data Pull"):
        query = f"""
        SELECT 
            T.hshd_num, T.basket_num, T.purchase_, T.product_num, 
            P.department, P.commodity
        FROM Transactions T
        JOIN Households H ON T.hshd_num = H.hshd_num
        JOIN Products P ON T.product_num = P.product_num
        WHERE T.hshd_num = {hshd}
        ORDER BY T.hshd_num, T.basket_num, T.purchase_, T.product_num
        """
        pull = pd.read_sql(query, conn)
        st.dataframe(pull)

# ------------------ BASKET ML ------------------
elif menu == "Basket ML" and st.session_state.logged_in:
    st.subheader("🔮 Predict Top 5 Commodities within a Category")

    # Load and merge data
    transactions = pd.read_sql("SELECT * FROM Transactions", conn)
    products = pd.read_sql("SELECT * FROM Products", conn)
    households = pd.read_sql("SELECT * FROM Households", conn)

    # Standardize column names
    for df in [transactions, products, households]:
        df.columns = df.columns.str.lower()

    df = transactions.merge(products, on="product_num").merge(households, on="hshd_num")

    # Clean & encode
    df["loyalty_flag"] = df["l"].map({"Y": 1, "N": 0})
    df["hh_size"] = df["hh_size"].astype(str).str.extract('(\d+)').fillna(0).astype(int)
    df["children"] = df["children"].astype(str).str.extract('(\d+)').fillna(0).astype(int)

    df = df.dropna(subset=["brand_ty", "natural_organic_flag", "department", "commodity"])

    # User selects category (department)
    st.markdown("### 🧭 Select a Product Category to Explore")
    selected_dept = st.selectbox("Choose a Department", df["department"].unique())

    # Filter only selected category
    df_dept = df[df["department"] == selected_dept]

    # Encode features
    features = df_dept[["hh_size", "children", "loyalty_flag", "brand_ty", "natural_organic_flag"]].copy()
    for col in ["brand_ty", "natural_organic_flag"]:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))

    # Set target = commodity
    df_dept["target"] = df_dept["commodity"]

    # Build and train model
    model = RandomForestClassifier()
    model.fit(features, df_dept["target"])

    # Predict top 5 commodities
    pred_probs = pd.DataFrame(model.predict_proba(features), columns=model.classes_)
    top5 = pred_probs.sum().sort_values(ascending=False).head(5)

    st.markdown(f"### 🎯 Top 5 Predicted Commodities in *{selected_dept}* Category")
    for i, (commodity, prob) in enumerate(top5.items(), 1):
        st.markdown(f"*{i}. {commodity}* — {round(prob * 100 / top5.sum(), 2)}%")

    # Feature Importance
    st.markdown("---")
    st.subheader("📊 Feature Influence on Predictions")
    feature_imp = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=True)
    st.bar_chart(feature_imp)


elif menu == "Churn Prediction" and st.session_state.logged_in:
    st.subheader("📉 Churn Prediction — At-Risk Households")

    # Load data from Azure SQL
    households = pd.read_sql("SELECT * FROM Households", conn)
    transactions = pd.read_sql("SELECT * FROM Transactions", conn)

    # Standardize column names
    households.columns = households.columns.str.strip().str.lower()
    transactions.columns = transactions.columns.str.strip().str.lower()

    # Clean hh_size and children
    def clean_numeric_column(series):
        return (
            series.astype(str)
            .str.replace("null", "0", case=False)
            .str.replace("5+", "5", case=False)
            .str.replace("3+", "3", case=False)
            .str.replace("None", "0", case=False)
            .str.extract("(\d+)")
            .fillna(0)
            .astype(int)
        )

    households["hh_size"] = clean_numeric_column(households["hh_size"])
    households["children"] = clean_numeric_column(households["children"])
    households["loyalty_flag"] = households["l"].map({"Y": 1, "N": 0}).fillna(0)

    # Churn Target: customers who spent < median spend
    total_spend = transactions.groupby("hshd_num")["spend"].sum().reset_index()
    total_spend.columns = ["hshd_num", "total_spend"]
    merged = households.merge(total_spend, on="hshd_num", how="left").fillna(0)
    median_spend = merged["total_spend"].median()
    merged["churn"] = (merged["total_spend"] < median_spend).astype(int)

    # Features
    features = merged[["hh_size", "children", "loyalty_flag"]]
    target = merged["churn"]

    # Train Model
    model = RandomForestClassifier()
    model.fit(features, target)

    # Prediction Summary
    churn_risk = pd.DataFrame({
        "Household": merged["hshd_num"],
        "Total Spend": merged["total_spend"],
        "Churn Risk (1=Yes)": model.predict(features)
    })

    st.markdown("### 🔍 Churn Risk Table")
    st.dataframe(churn_risk[churn_risk["Churn Risk (1=Yes)"] == 1].sort_values(by="Total Spend"))

    # Feature Importance
    st.markdown("### 🔎 Feature Importance")
    imp = pd.Series(model.feature_importances_, index=features.columns).sort_values()
    st.bar_chart(imp)