import requests

class LLMClient:
    """Supports local Ollama and OpenAI-compatible cloud Chat Completions."""
    def __init__(self, config):
        self.config = config

    def status(self):
        provider = self.config.LLM_PROVIDER
        if provider == "off":
            return {"provider": "off", "ready": False, "message": "LLM is disabled."}
        if provider in ("auto", "ollama"):
            try:
                r = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=3)
                r.raise_for_status()
                models = [m.get("name", "") for m in r.json().get("models", [])]
                if self.config.OLLAMA_MODEL in models or any(x.startswith(self.config.OLLAMA_MODEL.split(":")[0] + ":") for x in models):
                    return {"provider": "ollama", "ready": True, "message": f"Local model ready: {self.config.OLLAMA_MODEL}"}
                if provider == "ollama":
                    return {"provider": "ollama", "ready": False, "message": f"Ollama is running, but {self.config.OLLAMA_MODEL} is not downloaded."}
            except requests.RequestException:
                if provider == "ollama":
                    return {"provider": "ollama", "ready": False, "message": "Ollama is not reachable. Start Ollama and try again."}
        if provider in ("auto", "cloud"):
            if self.config.CLOUD_API_KEY:
                return {"provider": "cloud", "ready": True, "message": f"Cloud API configured: {self.config.CLOUD_MODEL}"}
            if provider == "cloud":
                return {"provider": "cloud", "ready": False, "message": "Add CLOUD_API_KEY to your .env file."}
        return {"provider": "none", "ready": False, "message": "No LLM is currently available. Rule-based analysis still works."}

    def _ollama(self, messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        conversation = [m for m in messages if m["role"] != "system"]
        response = requests.post(
            f"{self.config.OLLAMA_BASE_URL}/api/chat",
            json={"model": self.config.OLLAMA_MODEL, "messages": ([{"role": "system", "content": system}] if system else []) + conversation, "stream": False},
            timeout=self.config.LLM_TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _cloud(self, messages):
        response = requests.post(
            f"{self.config.CLOUD_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.CLOUD_API_KEY}", "Content-Type": "application/json"},
            json={"model": self.config.CLOUD_MODEL, "messages": messages, "temperature": 0.3},
            timeout=self.config.LLM_TIMEOUT)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def generate(self, messages):
        provider = self.config.LLM_PROVIDER
        if provider == "off":
            raise RuntimeError("LLM is disabled in configuration.")
        errors = []
        choices = [provider] if provider in ("ollama", "cloud") else ["ollama", "cloud"]
        for choice in choices:
            try:
                if choice == "ollama":
                    return self._ollama(messages), "ollama"
                if choice == "cloud":
                    if not self.config.CLOUD_API_KEY:
                        raise RuntimeError("Cloud API key is not configured.")
                    return self._cloud(messages), "cloud"
            except Exception as exc:
                errors.append(f"{choice}: {exc}")
        raise RuntimeError("; ".join(errors) or "No LLM provider is available.")
