from fastapi import FastAPI, Request,Depends, HTTPException,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os

import openai
from openai import OpenAI

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


api_key = os.getenv("OPENAI_API_KEY")
# api_key = os.getenv("GROQ_API_KEY")

agent = Agent(
    name="Basic Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key,max_tokens=100),
    # model=Groq(id="llama-3.3-70b-versatile",api_key=api_key),
    tools=[GoogleSearchTools()],
    # tools=[DuckDuckGoTools()],
    add_history_to_messages=True,
    num_history_responses=5,
    add_datetime_to_instructions=True,
    markdown=True,
)


question="What are the latest developments in AI?"

# result=agent.run(question)
# print("result.content: ",result.content)

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



@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = "temp_audio.webm"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    with open(temp_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    return {"transcription": transcription.text}

# @app.post("/run-agent")
# async def run_agent(prompt: Prompt):
#     try:
#         run = agent.run(prompt.message)
#         return {"response": run.content}
#     except Exception as e:
#         return {"error": str(e)}
    

