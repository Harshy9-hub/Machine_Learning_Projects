import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# LOAD MODEL + DATA
# -----------------------------------------
model = joblib.load("random_forest_price_model.pkl")
df = pd.read_csv("C:/Users/DHA01/Downloads/Harsh/ML/Data/amazon_all_electronics_data.csv")


# Clean & Feature Engineering (same as training time)
df["Availability"] = df["Availability"].map({"In Stock": 1, "Out of Stock": 0})
df["Name_Length"] = df["Product_Name"].apply(len)
df["Word_Count"] = df["Product_Name"].apply(lambda x: len(x.split()))
df["Category"] = df["Product_Name"].apply(lambda x: x.split()[0])

# Label Encoding same as training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Category_Encoded"] = le.fit_transform(df["Category"])

# STREAMLIT UI

st.title("📱 Product Price Predictor")
st.write("Select a product and predict its price using your ML model")

# Dropdown for product selection
product_list = df["Product_Name"].unique()
selected_product = st.selectbox("Select a Product", product_list)

# Fetch product row
row = df[df["Product_Name"] == selected_product].iloc[0]

# Show auto-filled details
st.subheader("📌 Product Details (Auto-filled)")
st.write(f"**Category:** {row['Category']}")
st.write(f"**Rating:** {row['Rating']}")
st.write(f"**Review Count:** {row['Review_Count']}")
availability_text = "In Stock" if row["Availability"] == 1 else "Out of Stock"
st.write(f"**Availability:** {availability_text}")
st.write(f"**Name Length:** {row['Name_Length']} characters")
st.write(f"**Word Count:** {row['Word_Count']} words")
st.write(f"**Category Encoded:** {row['Category_Encoded']}")

# Prepare input for prediction
input_data = pd.DataFrame({
    "Rating": [row["Rating"]],
    "Review_Count": [row["Review_Count"]],
    "Availability": [row["Availability"]],
    "Name_Length": [row["Name_Length"]],
    "Word_Count": [row["Word_Count"]],
    "Category_Encoded": [row["Category_Encoded"]],
})

# Predict button
if st.button("Predict Price 💰"):
    price = model.predict(input_data)[0]
    st.success(f"Estimated Price: $ {price:,.2f}")
    