# ============================================
# Nutrition AI Agent
# LangChain + RAG + Gemini + FastAPI
# ============================================

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import os
from PIL import Image
import io

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# RAG
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ==============================
# Gemini Model
# ==============================

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)

# ==============================
# FastAPI
# ==============================

app = FastAPI(title="Nutrition AI Agent")

# ==============================
# RAG Knowledge Base
# ==============================

loader = TextLoader("nutrition_knowledge.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# ==============================
# Tools
# ==============================

# -------- Calories Tool --------

def calorie_calculator(data: str):

    try:
        weight, height, age = map(float, data.split(","))

        bmr = 10*weight + 6.25*height - 5*age + 5

        return f"Estimated daily calories: {int(bmr*1.2)} kcal"

    except:
        return "Format: weight,height,age"


# -------- Meal Plan Tool --------

def meal_planner(goal: str):

    prompt = f"""
    Create a daily nutrition plan for a person with goal: {goal}
    include breakfast lunch dinner snacks
    """

    return llm.invoke(prompt).content


# -------- Weight Tracker --------

user_weights = []

def weight_tracker(weight: str):

    user_weights.append(float(weight))

    avg = sum(user_weights)/len(user_weights)

    return f"Weight logged. Current average weight: {avg}"


# -------- RAG Nutrition Tool --------

def nutrition_rag(question: str):

    docs = retriever.get_relevant_documents(question)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer using the nutrition context.

    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt).content


# ==============================
# Image Analysis Tool
# ==============================

def analyze_food_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes))

    prompt = """
    Identify the food in the image and estimate calories.
    """

    response = llm.invoke(
        [prompt, img]
    )

    return response.content


# ==============================
# Agent Tools
# ==============================

tools = [

    Tool(
        name="Calorie Calculator",
        func=calorie_calculator,
        description="calculate calories using weight,height,age"
    ),

    Tool(
        name="Meal Planner",
        func=meal_planner,
        description="generate diet plan"
    ),

    Tool(
        name="Weight Tracker",
        func=weight_tracker,
        description="track weight progress"
    ),

    Tool(
        name="Nutrition Knowledge",
        func=nutrition_rag,
        description="answer nutrition questions"
    )

]

memory = ConversationBufferMemory()

agent = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True
)

# ==============================
# Request Models
# ==============================

class ChatRequest(BaseModel):
    message: str


# ==============================
# API Endpoints
# ==============================

@app.post("/chat")

def chat(req: ChatRequest):

    response = agent.run(req.message)

    return {"response": response}


# -----------------------------

@app.post("/analyze-food")

async def analyze_food(file: UploadFile = File(...)):

    image_bytes = await file.read()

    result = analyze_food_image(image_bytes)

    return {"analysis": result}


# -----------------------------

@app.post("/log-weight")

def log_weight(weight: float):

    result = weight_tracker(str(weight))

    return {"result": result}


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
