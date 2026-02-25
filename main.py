 from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

app = FastAPI(title="AI Track Planning Agent")

# ==============================
# تحميل موديل Open Source
# ==============================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=700
)

llm = HuggingFacePipeline(pipeline=pipe)

# ==============================
# Prompt
# ==============================

plan_prompt = PromptTemplate(
    input_variables=["track", "level", "hours", "goal"],
    template="""
أنت مستشار تقني محترف.

المعطيات:
التراك: {track}
المستوى: {level}
عدد الساعات يوميًا: {hours}
الهدف: {goal}

أنشئ خطة تعلم مفصلة لمدة 3 شهور.
قسّمها بأسابيع.
اذكر:
- المهارات المطلوبة
- مصادر تعلم مجانية
- مشاريع عملية
- milestones واضحة
"""
)

plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

# ==============================
# Session Storage
# ==============================

sessions = {}

class StartRequest(BaseModel):
    user_id: str
    track: str

class AnswerRequest(BaseModel):
    user_id: str
    answer: str

# ==============================
# Endpoints
# ==============================

@app.get("/")
def home():
    return {"message": "AI Track Agent Running 🚀"}

@app.post("/start")
def start_agent(data: StartRequest):
    sessions[data.user_id] = {
        "track": data.track,
        "step": 1
    }

    return {"question": "مستواك إيه؟ (مبتدئ - متوسط - متقدم)"}

@app.post("/answer")
def answer_question(data: AnswerRequest):

    user = sessions.get(data.user_id)

    if not user:
        return {"error": "ابدأ الأول من /start"}

    step = user["step"]

    if step == 1:
        user["level"] = data.answer
        user["step"] = 2
        return {"question": "كام ساعة تقدر تذاكر يوميًا؟"}

    elif step == 2:
        user["hours"] = data.answer
        user["step"] = 3
        return {"question": "هدفك إيه من التراك ده؟"}

    elif step == 3:
        user["goal"] = data.answer

        plan = plan_chain.run({
            "track": user["track"],
            "level": user["level"],
            "hours": user["hours"],
            "goal": user["goal"]
        })

        sessions.pop(data.user_id)

        return {
            "final_plan": plan
        }