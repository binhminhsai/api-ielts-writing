from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import build_prompt
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import uvicorn

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    print("❌ Lỗi: Không tìm thấy OPENAI_API_KEY. Kiểm tra lại file .env của bạn.")
    exit()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

app = FastAPI(
    title="Wispace AI Writing Assistant",
    description="API grading, feedback & vocabulary tool for IELTS Writing Task using OpenAI Response API.",
)

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]
app.add_middleware (
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,    
    allow_methods=["*"],      
    allow_headers=["*"],      
)

class WritingTask1Input(BaseModel):
    question: str
    url: str
    topic: str
    level: str

class GradingTask1Input(BaseModel):
    question: str
    url: str
    topic: str
    essay: str

class WritingTask2Input(BaseModel):
    question: str
    topic: str
    level: str

class GradingTask2Input(BaseModel):
    question: str
    topic: str
    essay: str

# --- ENDPOINT 1: WRITING ASSISTANT TASK 1---
@app.post("/writing-assistant-task1", tags=["AI Assistants"])
async def writing_assistant_task1(input_data: WritingTask1Input):
    """Tạo nội dung bài viết dựa trên thông tin đầu vào (text + image)."""
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="writing_assistant_task1")

        input_json_string = input_data.model_dump_json(indent=2)
        user_prompt_text = (
            "### Task:\n"
            "Now do the same for the following input:\n"
            f"{input_json_string}\n\n"
            "Return JSON output in the same structure."
        )

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_text},
                        {"type": "input_image", "image_url": input_data.url}
                    ]
                }
            ],
            temperature=0.2
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: VOCAB SUGGESTION ---
@app.post("/vocab-suggestion-task1", tags=["AI Assistants"])
async def vocab_suggestion_task1(input_data: WritingTask1Input):
    """Gợi ý từ vựng nâng cao (advanced vocabulary) dựa trên thông tin đầu vào (question + image + band level)."""
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="vocab_suggestion_task1")

        input_json_string = input_data.model_dump_json(indent=2)
        user_prompt_text = (
            "### Task:\n"
            "Now do the same for the following input:\n"
            f"{input_json_string}\n\n"
            "Return JSON output in the same structure."
        )


        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_text},
                        {"type": "input_image", "image_url": input_data.url}
                    ]
                }
            ],
            temperature=0.3,
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 3: GRADING & FEEDBACK ---
@app.post("/grading-feedback-task1", tags=["AI Assistants"])
async def grading_feedback_task1(input_data: GradingTask1Input):
    """
    Phân tích bài viết (essay) của người dùng, chấm điểm theo tiêu chí IELTS Writing Task 1,
    đồng thời cung cấp phản hồi chi tiết (feedback).
    """
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="grading_feedback_task1")

        input_json_string = input_data.model_dump_json(indent=2)
        user_prompt_text = (
            "### Task:\n"
            "Now do the same for the following input:\n"
            f"{input_json_string}\n\n"
            "Return JSON output in the same structure."
        )

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_text},
                        {"type": "input_image", "image_url": input_data.url}
                    ]
                }
            ],
            temperature=0.3,
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 4: WRITING ASSISTANT TASK 2 ---
@app.post("/writing-assistant-task2", tags=["AI Assistants"])
async def writing_assistant_task2(input_data: WritingTask2Input):
    """Tạo nội dung bài viết dựa trên thông tin đầu vào (question + topic + band level)."""
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="writing_assistant_task2")
        input_json_string = input_data.model_dump_json(indent=2)
        user_prompt_text = (
            "### Task:\n"
            "Now do the same for the following input:\n"
            f"{input_json_string}\n\n"
            "Return JSON output in the same structure."
        )

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_text}
                    ]
                }
            ],
            temperature=0.3,
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 5: VOCAB SUGGESTION TASK 2 ---
@app.post("/vocab-suggestion-task2", tags=["AI Assistants"])
async def vocab_suggestion_task2(input_data: WritingTask2Input):
    """"Gợi ý từ vựng nâng cao (advanced vocabulary) dựa trên thông tin đầu vào (question + band level)."""
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="vocab_suggestion_task2")

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are an IELTS Vocabulary assistant. "
                                "Analyze the given chart from url and suggest academic words and collocations "
                                "in JSON format only — categories like 'Verbs for Trends', "
                                "'Nouns for Changes', 'Adjectives/Adverbs for Degree', etc."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ],
                },
            ],
            temperature=0.2,
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 6: GRADING & FEEDBACK TASK 2 ---
@app.post("/grading-feedback-task2", tags=["AI Assistants"])
async def grading_feedback_task2(input_data: GradingTask2Input):
    """
    Phân tích bài viết (essay) của người dùng, chấm điểm theo tiêu chí IELTS Writing Task 2,
    đồng thời cung cấp phản hồi chi tiết (feedback).
    """
    try:
        prompt = build_prompt(input_data.dict(), prompt_name="grading_feedback_task2")

        input_json_string = input_data.model_dump_json(indent=2)
        user_prompt_text = (
            "### Task:\n"
            "Now do the same for the following input:\n"
            f"{input_json_string}\n\n"
            "Return JSON output in the same structure."
        )

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_text}
                    ]
                }
            ],
            temperature=0.3,
        )

        output_text = response.output[0].content[0].text.strip()
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            result = {"raw_output": output_text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RUN SERVER ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 1000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
