"""HTTP API: /health and POST /ask."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rocm_agent import __version__
from rocm_agent.answer import Answer, answer_question

app = FastAPI(title="ROCm Docs Agent", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(pattern=r"\S")  # rejects an empty or blank question with a 422
    k: int | None = Field(default=None, ge=1)
    agent: bool = False  # let the model call tools instead of one fixed retrieval


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ask")
def ask(request: AskRequest) -> Answer:
    if request.agent:
        from rocm_agent.agent import run_agent

        return run_agent(request.question)
    return answer_question(request.question, request.k)
