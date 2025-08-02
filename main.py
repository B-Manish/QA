from fastapi import FastAPI, Request,Depends, HTTPException,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools import tool
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os
from litellm import completion
import tempfile
import openai
from openai import OpenAI
from typing import Literal

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GROQ_API_KEY")


# category: Literal["DSA", "System Design", "Frontend", "Backend", "CS Fundamentals", "Other"] = "Other"
# Category: {category}
@tool(name="answer_interview_question")
def answer_interview_question(question: str):
    """
    Answer technical interview questions for developers across various categories.

    Parameters:
    - question: The actual technical question being asked.
    - category: (Optional) The type of question. Helps tailor the response.
    """

    print("inside answer_interview_question called")

    prompt = f"""
    You are a helpful technical interviewer.


    - Do not use technical jargon.
    - Do not include code.
    - Use basic English to answer the question.
    - Use clear, original explanations.
    - Keep it short and easy to understand.
    - Avoid sounding like content from the internet or tutorials.

    Question: "{question}"

    Respond in Markdown. Do not include headings or extra formatting — just the plain English answer.
    """

    
    response = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

agent = Agent(
    name="Interview Agent",
    instructions="You are an expert interviewer helping users answer technical developer interview questions.If the user asks a developer interview question, use the `answer_interview_question` tool to respond.If the tool provides a good response, return it directly.",
    # model=OpenAIChat(id="gpt-3.5-turbo", api_key=api_key),
    model=Groq(id="llama-3.3-70b-versatile",api_key=api_key),
    tools=[GoogleSearchTools(),answer_interview_question],
    add_history_to_messages=True,
    num_history_responses=5,
    add_datetime_to_instructions=True,
    markdown=True,
)


question="Explain reacts virtual dom"

result=agent.run(question)
print("result: ",result.content)

def convert_audio_to_text():
    audio_path = "audio.mp3"

    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
        transcribed_text = transcription.text
        print("🎙️ Transcribed Text:", transcribed_text)

# convert_audio_to_text()



# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     temp_path = "temp_audio.webm"
#     with open(temp_path, "wb") as f:
#         f.write(await file.read())

#     with open(temp_path, "rb") as f:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=f,
#             language="en" # translate any language to english
#         )

#     return {"transcription": transcription.text}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = "temp_audio.webm"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    with open(temp_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            # language="en" # translate any language to english
        )

    return {"transcription": transcription.text}
    

