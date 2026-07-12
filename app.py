import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1"
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


SYSTEM_PROMPT = """
You answer questions about images.

Rules:
- Return ONLY JSON.
- Output format:

{
  "answer":"..."
}

- answer MUST always be a string.
- If the answer is numeric, return only the number as a string.
- Do not include currency symbols.
- Do not include units.
- No markdown.
- No explanation.
"""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/answer-image")
def answer_image(req: ImageRequest):

    image_url = f"data:image/png;base64,{req.image_base64}"

    try:

        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": req.question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
        )

        result = json.loads(
            response.choices[0].message.content
        )

        answer = result.get("answer", "")

        if answer is None:
            answer = ""

        return {
            "answer": str(answer).strip()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )