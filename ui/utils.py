import os
from typing import Iterable

import httpx


def get_backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000")


def post_json(path: str, payload: dict) -> dict:
    base = get_backend_url().rstrip("/")
    url = f"{base}{path}"
    with httpx.Client(timeout=200.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def stream_json(path: str, payload: dict) -> Iterable[str]:
    base = get_backend_url().rstrip("/")
    url = f"{base}{path}"
    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text(chunk_size=1024):
            if chunk:
                yield chunk
