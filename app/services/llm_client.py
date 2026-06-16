from typing import Protocol

import httpx


class LLMClient(Protocol):
    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer from a question and retrieved context."""


class MockLLMClient:
    """Local fallback LLM for demos and tests without external API calls."""

    def generate_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return (
                "Mock answer: I could not find relevant context in the uploaded documents. "
                "Upload a document first or ask a question covered by the indexed content."
            )

        compact_context = " ".join(context.split())
        preview = compact_context[:700].rstrip()
        return (
            "Mock answer: based on the retrieved document context, the answer is likely "
            f"related to: {preview}"
        )


class OpenAICompatibleLLMClient:
    """Chat-completions client for OpenAI-compatible LLM providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_answer(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful RAG assistant. Answer only from the provided context. "
                    "If the answer is not in the context, say that the documents do not contain "
                    "enough information. Keep the answer concise and cite sources indirectly by "
                    "referring to the provided source labels."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{question}\n\n"
                    "Answer:"
                ),
            },
        ]
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return str(content).strip()
