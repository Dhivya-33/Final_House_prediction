import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.title("🏠 Realistic Housing Price Prediction App")
st.write("Predict house prices based on location and features using Machine Learning!")

file_name = 'realistic_housing_data.xlsx'

try:
    df = pd.read_excel(file_name)
    st.success("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ File not found! Please make sure 'realistic_housing_data.xlsx' is in the same folder.")
    st.stop()

with st.expander("📊 View Dataset Summary"):
    st.write("**Shape of the Dataset:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())
    st.write("**Statistical Summary:**")
    st.write(df.describe())
with st.expander("📈 Correlation Heatmap"):
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    st.pyplot(plt)
if "price" not in df.columns:
    st.error("The dataset must contain a column named 'price'.")
    st.stop()

X = df.drop("price", axis=1)
y = df["price"]


categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("scale", StandardScaler(), numerical_features)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** ₹{mae:,.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** ₹{rmse:,.2f}")
st.write(f"**R² Score:** {r2:.2f}")

st.subheader("🏡 Predict House Price")

user_input = {}
for col in X.columns:
    if col in categorical_features:
        user_input[col] = st.selectbox(f"Select {col}", df[col].dropna().unique())
    else:
        user_input[col] = st.number_input(f"Enter {col}", min_value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("🔮 Predict Price"):
    predicted_price = pipeline.predict(input_df)[0]
    st.success(f"🏠 Predicted House Price: ₹{predicted_price:,.2f}")

with st.expander("📉 Actual vs Predicted Prices"):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    st.pyplot(plt)

st.caption("Developed by Dhivi ✨ | Powered by Streamlit & Scikit-learn")
