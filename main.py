from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Simple Nutrition Prototype")

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    response = llm.invoke(req.message)
    return {"response": response.content}

# Calculate calories endpoint
@app.post("/calories")
def calories(data: str):
    try:
        weight, height, age = map(float, data.split(","))
        bmr = 10*weight + 6.25*height - 5*age + 5
        return {"calories": int(bmr*1.2)}
    except:
        return {"error": "Format: weight,height,age"}

# Meal plan endpoint
@app.post("/meal-plan")
def meal_plan(goal: str):
    prompt = f"Create a daily nutrition plan for a person with goal: {goal}. Include breakfast, lunch, dinner, snacks."
    response = llm.invoke(prompt)
    return {"meal_plan": response.content}
