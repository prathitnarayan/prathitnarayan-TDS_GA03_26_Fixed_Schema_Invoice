import base64
import io
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(req: Request):

    image_data = req.image_base64

    # Remove data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img = Image.open(
        io.BytesIO(base64.b64decode(image_data))
    )
    prompt = f"""
Answer ONLY the question.

Question:
{req.question}

Rules:
- Return only the answer.
- If numeric, return only the number.
- No explanation.
"""

    response = model.generate_content(
        [prompt, img]
    )

    return {
        "answer": response.text.strip()
    }