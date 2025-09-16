import os
import httpx


def get_backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000")


def post_json(path: str, payload: dict) -> dict:
    base = get_backend_url().rstrip("/")
    url = f"{base}{path}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
