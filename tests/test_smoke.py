from fastapi.testclient import TestClient

from rocm_agent import config
from rocm_agent.api import app


def test_config_loads():
    s = config.load()
    assert s.corpus.name == "rocm"
    assert s.corpus.subdir == "docs"


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
