# Basic packages
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import xgboost as xgb
model1 = xgb.XGBClassifier()



# load the model
import joblib
xgb = joblib.load("XGB.pkl")


# Title
st.title("🚢 Titanic Data Portal")
st.write("Predicting Titanic survival using XGB model")


st.markdown(r"""
    <style>
    .stApp {
        background-image: url("https://i.ibb.co/VWchZsq9/mymind-Kn-UX9qt-R-4-E-unsplash.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", 
     unsafe_allow_html=True
)


# Load dataset
try:
    df = pd.read_csv("titanic_csv.csv")
    st.success("✨Titanic Dataset loaded successfully")
except FileNotFoundError:
    st.error("🧐 Titanic dataset file not found")
    st.stop()

# Sidebar navigation
section = st.sidebar.radio("Select Dataset Section",
                           ["Dataset Preview", "Dataset Information", "Numerical Summary"])

# Sidebar sections
if section == "Dataset Preview":
    view_option = st.sidebar.radio("To view the dataset, select show", ["Hide", "Show"])
    if view_option == "Show":
        st.sidebar.subheader("💢Dataset Preview")
        st.sidebar.dataframe(df.head())

elif section == "Dataset Information":
    st.sidebar.subheader("⚡Dataset Information")
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Number of Rows", value=df.shape[0])
    col2.metric(label="Number of Columns", value=df.shape[1])

elif section == "Numerical Summary":
    with st.sidebar.expander("📊Summary of Numerical Columns", expanded=False):
        st.write(df.describe())

# Chatbot Section
st.subheader("👀Smart ChatBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chatbot_response(text):
    text = text.lower()
    if "hi" in text:
        return "Hello! I’m your Titanic Data Assistant 🚢"
    elif "who" in text:
        return "I’m built by Diksha to help analyze Titanic data!"
    elif "titanic" in text:
        return "The Titanic dataset records passengers' information — age, class, gender, fare, and survival."
    else:
        return "Sorry, I don’t have an answer for that yet."

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = chatbot_response(user_input)
    st.session_state.chat_history.append(("You : " + user_input, "Bot : " + response))
    st.rerun()

for user_msg, bot_msg in st.session_state.chat_history:
    st.write(user_msg)
    st.write(bot_msg)

if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Manual Login Section
credentials = {
    "director": "director123",
    "manager": "manager123",
    "trainer": "trainer123",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔒Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in credentials and credentials[username] == password:
            st.success(f"😁Welcome, {username}!")
            st.session_state.logged_in = True
        else:
            st.error("🧠Invalid username or password")

# Show visual FAQs if logged in
if st.session_state.logged_in:
    st.write("🤖You now have access to the Titanic Analysis App!")

    st.subheader("📊Titanic Visual FAQs")

    # Q1: Age Distribution
    with st.expander("Q1: What is the Age Distribution of passengers?"):
        st.write("Most passengers were between 20–40 years old.")
        fig_age = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    # Q2: Gender Count
    with st.expander("Q2: How many male and female passengers were there?"):
        st.write("There were more male passengers than female.")
        fig_sex = px.bar(df['Sex'].value_counts(),  
                 x=df['Sex'].value_counts().index,
                 y=df['Sex'].value_counts().values,
                 title="Gender Count",
                 labels={'x': 'Gender', 'y': 'Count'})

    # Q3: Survival by Gender
    with st.expander("Q3: What is the survival rate by gender?"):
        st.write("Females had a higher survival rate compared to males.")
        fig_survival_gender = px.bar(df, x='Sex', color='Survived',
                                     barmode='group', title="Survival Count by Gender")
        st.plotly_chart(fig_survival_gender, use_container_width=True)

    # Q4: Passenger Class Distribution
    with st.expander("Q4: How many passengers were in each class?"):
        st.write("Most passengers traveled in 3rd class.")
        fig_class = px.pie(df, names='Pclass', title="Passenger Class Distribution")
        st.plotly_chart(fig_class, use_container_width=True)

    # Q5: Survival by Passenger Class
    with st.expander("Q5: What is the survival rate across passenger classes?"):
        st.write("Passengers in 1st class had higher survival rates.")
        fig_survival_class = px.bar(df, x='Pclass', color='Survived',
                                    barmode='group', title="Survival Count by Class")
        st.plotly_chart(fig_survival_class, use_container_width=True)

    # Q6: Fare Distribution
    with st.expander("Q6: What is the fare distribution of passengers?"):
        st.write("Most passengers paid between 0–100 in fare, with few paying much higher.")
        fig_fare = px.histogram(df, x="Fare", nbins=30, title="Fare Distribution")
        st.plotly_chart(fig_fare, use_container_width=True)



    # --- Input fields in two columns ---
    col1, col2 = st.columns(2)
    with col1:
        passenger_id = st.number_input("Passenger ID", min_value=1, max_value=1000, value=1)
        pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
        age = st.slider("Age", min_value=0, max_value=100, value=25)
        sibsp = st.slider("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.slider("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=7.25)

# ---------------------- Column 2 (Categorical Inputs) ----------------------
    with col2:
        name = st.text_input("Name", "John Doe")
        sex = st.selectbox("Sex", ["male", "female"])
        ticket = st.text_input("Ticket", "A/5 21171")
        cabin = st.text_input("Cabin", "C85")
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

            # ---------------------- Prepare Input Data ----------------------
    input_data = {
        "PassengerId": passenger_id,
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked
    }



    input_df = pd.DataFrame([input_data])

    # ---------------------- Derived Features (Match Training Data) ----------------------
    input_df["Alone"] = (input_df["SibSp"] + input_df["Parch"] == 0).astype(int)
    input_df["Title"] = input_df["Name"].apply(lambda n: n.split('.')[0].split(' ')[-1])

    # ---------------------- Encoding ----------------------
    input_df["Sex"] = input_df["Sex"].map({"male": 0, "female": 1})
    input_df["Embarked"] = input_df["Embarked"].map({"C": 0, "Q": 1, "S": 2})





    # ----- Predict Button -----
    if st.button("🔮 Predict"):
        # Define categorical columns
        cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']

            # Display result

        st.success("✅ The client is **likely to subscribe** to the term deposit.")
    
        st.error("❌ The client is **unlikely to subscribe** to the term deposit.")  

st.caption("Developed by Diksha 💻 | XGB Model | Streamlit App")  # Footer            

           
