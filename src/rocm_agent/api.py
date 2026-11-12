"""HTTP API. Only /health exists until the agent is built on Saturday."""

from fastapi import FastAPI

from rocm_agent import __version__

app = FastAPI(title="ROCm Docs Agent", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
