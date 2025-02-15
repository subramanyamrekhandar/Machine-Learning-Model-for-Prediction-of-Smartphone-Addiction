import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu

# Load dataset
file_path = "dataset.xlsx"  # Update with your dataset path
df = pd.read_excel(file_path, sheet_name="Form Responses 1")

# Select relevant features for prediction
selected_columns = [
    "AGE ?", 
    "When your phone rings, beeps, buzzes, do you feel an intense urge to check for texts, tweets, or emails, updates, etc.?",
    "Do you find yourself mindlessly checking your phone many times a day even when you know there is likely nothing new or important to see?",
    "Do you feel reluctant/uncomfortable to be without your smartphone, even for a short time?",
    "Do you feel your use of your cell phone actually decreases your productivity at times?"
]
df = df[selected_columns]

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define target variable
X = df.drop(columns=["Do you feel your use of your cell phone actually decreases your productivity at times?"])
y = df["Do you feel your use of your cell phone actually decreases your productivity at times?"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Smartphone Addiction Prediction", layout="centered")

# Sidebar Navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Prediction"])
with st.sidebar:
    page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction"],
        icons=["house", "robot"],
        menu_icon="cast",
        default_index=0,
    )

if page == "Home":
   
    st.title("Prediction of Smartphone Addiction by Using ML")
    st.write("Developed by Subramanyam Rekhandar")
    st.write("This app predicts smartphone addiction based on user inputs and survey data.")
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.image("banner.jpg", use_container_width=True)  # Add a banner image

elif page == "Prediction":
    st.title("Prediction of Smartphone Addiction by Using ML")
    st.sidebar.header("User Input")
    age = st.sidebar.slider("Age", 10, 60, 25)
    phone_urge = st.sidebar.selectbox("Do you feel an intense urge to check notifications?", label_encoders[selected_columns[1]].classes_)
    mindless_check = st.sidebar.selectbox("Do you check your phone mindlessly?", label_encoders[selected_columns[2]].classes_)
    uncomfortable_without_phone = st.sidebar.selectbox("Do you feel uncomfortable without your smartphone?", label_encoders[selected_columns[3]].classes_)
    
    if st.sidebar.button("Predict"):
        # Convert inputs to model format
        input_data = np.array([
            age,
            label_encoders[selected_columns[1]].transform([phone_urge])[0],
            label_encoders[selected_columns[2]].transform([mindless_check])[0],
            label_encoders[selected_columns[3]].transform([uncomfortable_without_phone])[0]
        ]).reshape(1, -1)
        input_data = scaler.transform(input_data)
        
        # Predict addiction level
        prediction = model.predict(input_data)
        predicted_label = label_encoders[selected_columns[4]].inverse_transform(prediction)[0]
        st.write("### ⚠️ Prediction of Smartphone Addiction: ", predicted_label)
