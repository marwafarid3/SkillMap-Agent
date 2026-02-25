# tech_agent_full.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All

# ================================
# 1️⃣ إعداد LLM محلي
# ================================
llm = GPT4All(model="ggml-gpt4all-j-v1.3-groovy")  # أحدث نسخة GPT4All محلي

# ================================
# 2️⃣ إعداد قالب المحادثة
# ================================
plan_template = """
المستخدم مهتم بالتراك التكنولوجي: {track_name}
مستواه: {level}
عدد الساعات يوميًا: {hours_per_day}
هدفه: {goal}

اعمل خطة تعلم مخصصة له على حسب المعلومات دي.
"""

prompt = PromptTemplate(
    input_variables=["track_name", "level", "hours_per_day", "goal"],
    template=plan_template
)

agent_chain = LLMChain(llm=llm, prompt=prompt)

# ================================
# 3️⃣ دالة توليد الخطة
# ================================
def generate_learning_plan(track_name: str, level: str, hours_per_day: str, goal: str):
    return agent_chain.run(
        track_name=track_name,
        level=level,
        hours_per_day=hours_per_day,
        goal=goal
    )

# ================================
# 4️⃣ إعداد FastAPI
# ================================
app = FastAPI(title="Tech Learning Agent")

class UserInput(BaseModel):
    track_name: str
    level: str
    hours_per_day: str
    goal: str

@app.get("/")
def home():
    return {"message": "مرحبًا! استخدم /generate-plan لإنتاج خطة تعلم."}

@app.post("/generate-plan")
def generate_plan(user_input: UserInput):
    plan = generate_learning_plan(
        track_name=user_input.track_name,
        level=user_input.level,
        hours_per_day=user_input.hours_per_day,
        goal=user_input.goal
    )
    return {"learning_plan": plan}

# ================================
# 5️⃣ لتشغيل السيرفر مباشرة
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
