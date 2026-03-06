import streamlit as st
import pandas as pd
from PIL import Image
import os

# ================================
# LangChain Imports (NEW VERSION)
# ================================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA


# ================================
# Gemini API KEY
# ================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFcvpt-Fs_muflBT96HNZbw4c_9Axa0ik"


# ================================
# Load Gemini Model
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


# ================================
# Nutrition Knowledge Base (RAG)
# ================================
nutrition_data = [
    "Protein foods include chicken, eggs, fish, beans and yogurt.",
    "Healthy carbohydrates include oats, brown rice and sweet potatoes.",
    "Healthy fats include olive oil, avocado and nuts.",
    "Weight loss requires a calorie deficit.",
    "Drink at least 2 liters of water daily.",
    "Vegetables are low calorie and rich in nutrients.",
    "Protein supports muscle growth.",
    "Avoid processed food and excessive sugar."
]

docs = [Document(page_content=text) for text in nutrition_data]


splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)


# ================================
# Embeddings + Vector DB
# ================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever()


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# ================================
# Streamlit UI
# ================================
st.title("🥗 Nutrition AI Agent")


# ================================
# Sidebar Profile
# ================================
st.sidebar.header("User Profile")

weight = st.sidebar.number_input("Weight (kg)", 40, 200)
height = st.sidebar.number_input("Height (cm)", 140, 210)
age = st.sidebar.number_input("Age", 10, 80)

goal = st.sidebar.selectbox(
    "Goal",
    ["Lose Weight", "Maintain Weight", "Gain Muscle"]
)


# ================================
# Calories Calculator
# ================================
def calculate_calories(weight, height, age):

    bmr = 10 * weight + 6.25 * height - 5 * age + 5

    if goal == "Lose Weight":
        calories = bmr - 400
    elif goal == "Gain Muscle":
        calories = bmr + 300
    else:
        calories = bmr

    return int(calories)


if st.sidebar.button("🔥 Calculate Calories"):

    calories = calculate_calories(weight, height, age)

    st.sidebar.success(f"Daily Calories: {calories}")


# ================================
# Diet Plan Generator
# ================================
st.header("🥗 Generate Diet Plan")

if st.button("Generate Diet Plan"):

    prompt = f"""
    Create a healthy diet plan.

    Weight: {weight}
    Height: {height}
    Age: {age}
    Goal: {goal}

    Include:
    breakfast
    lunch
    dinner
    snacks
    """

    response = llm.invoke(prompt)

    st.write(response.content)


# ================================
# Food Image Analysis
# ================================
st.header("📷 Food Image Analysis")

uploaded_file = st.file_uploader("Upload food image")

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Food Image")

    prompt = "Describe this food and estimate calories."

    response = llm.invoke([prompt, image])

    st.write(response.content)


# ================================
# Weight Tracker
# ================================
st.header("📊 Weight Tracker")

if "weights" not in st.session_state:
    st.session_state.weights = []

new_weight = st.number_input("Enter today's weight")

if st.button("Add Weight"):

    st.session_state.weights.append(new_weight)


df = pd.DataFrame(
    st.session_state.weights,
    columns=["Weight"]
)

if not df.empty:

    st.line_chart(df)


# ================================
# Chat with AI (RAG)
# ================================
st.header("💬 Chat with Nutrition AI")

user_question = st.text_input(
    "Ask about nutrition..."
)

if user_question:

    answer = qa_chain.run(user_question)

    st.write(answer)
