from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import AIAgent


app = FastAPI(title="Direct Agricultural AI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AIAgent()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or prompt")


class ChatResponse(BaseModel):
    answer: str
    model: str
    mode: str


@app.get("/health")
def health():
    return {"status": "ok", "model": agent.model_name}


@app.get("/brief")
def brief():
    return {"brief": agent.research_brief()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = agent.chat(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    mode = "POP Bank direct answer" if agent._classify_query(request.message) == "pop_lookup" else "LLM response"
    return ChatResponse(answer=answer, model=agent.model_name, mode=mode)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)