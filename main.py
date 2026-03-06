import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# ======================
# LOAD API KEY
# ======================

load_dotenv()
API_KEY = os.getenv("AIzaSyAFcvpt-Fs_muflBT96HNZbw4c_9Axa0ik")

# ======================
# GEMINI MODEL
# ======================

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=API_KEY,
    temperature=0.2
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY
)

# ======================
# NUTRITION KNOWLEDGE
# ======================

knowledge = [
"Protein should be about 30% of daily calories",
"Fat should be about 25% of daily calories",
"Carbohydrates should be about 45% of daily calories",
"TDEE equals BMR multiplied by activity level",
"Weight loss requires calorie deficit"
]

vectorstore = FAISS.from_texts(knowledge, embeddings)

# ======================
# CALCULATIONS
# ======================

def calculate_bmr(weight, height, age, gender):

    if gender == "male":
        return (10*weight)+(6.25*height)-(5*age)+5
    else:
        return (10*weight)+(6.25*height)-(5*age)-161

def activity_factor(level):

    factors={
        "low":1.2,
        "medium":1.55,
        "high":1.9
    }

    return factors[level]

# ======================
# UI
# ======================

st.title("🥗 AI Nutrition Planner")

tab1,tab2=st.tabs(["Diet Generator","Nutrition Chat"])

# ======================
# DIET GENERATOR
# ======================

with tab1:

    goal=st.selectbox("Goal",["weight loss","muscle gain","maintenance"])

    weight=st.number_input("Weight",40,200)

    height=st.number_input("Height",120,220)

    age=st.number_input("Age",10,80)

    gender=st.selectbox("Gender",["male","female"])

    activity=st.selectbox("Activity",["low","medium","high"])

    if st.button("Generate Diet Plan"):

        bmr=calculate_bmr(weight,height,age,gender)

        tdee=bmr*activity_factor(activity)

        prompt=f"""
        Create a scientific 7 day diet plan.

        Goal: {goal}
        BMR: {bmr}
        TDEE: {tdee}
        """

        result=llm.invoke(prompt)

        st.subheader("Your Metrics")

        st.write("BMR:",round(bmr,2))
        st.write("TDEE:",round(tdee,2))

        st.subheader("Diet Plan")

        st.write(result.content)

# ======================
# CHAT
# ======================

with tab2:

    question=st.text_input("Ask nutrition question")

    if st.button("Ask AI"):

        docs=vectorstore.similarity_search(question,k=2)

        context="\n".join([d.page_content for d in docs])

        prompt=f"""
        Context:
        {context}

        Question:
        {question}
        """

        result=llm.invoke(prompt)

        st.write(result.content)
