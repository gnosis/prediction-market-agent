import requests


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    r = requests.get(f"{base_url}/api/tags")
    return r.status_code == 200
