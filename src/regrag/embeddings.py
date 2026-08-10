"""Voyage AI embeddings.

Anthropic does not serve an embeddings endpoint and points at Voyage as its
recommended provider, so the Claude API handles generation and Voyage handles
retrieval.
"""

import os
from typing import Any, Protocol

DEFAULT_MODEL = "voyage-3"
# Voyage caps documents per request; batching here rather than at the call site
# keeps the limit in one place.
DEFAULT_BATCH_SIZE = 128


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], model: str, input_type: str) -> Any: ...


class Embedder:
    def __init__(
        self,
        client: EmbeddingClient,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus chunks for indexing."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed(batch, input_type="document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question.

        Voyage embeds queries and documents into the same space but asymmetrically
        — passing input_type="document" here still returns a vector, just a worse
        one, so the mistake shows up as degraded recall rather than an error.
        """
        (vector,) = self._embed([text], input_type="query")
        return vector

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(
            texts, model=self._model, input_type=input_type
        )
        return response.embeddings


def build_embedder(model: str = DEFAULT_MODEL) -> Embedder:
    """Embedder backed by the real Voyage client, keyed from the environment."""
    import voyageai

    return Embedder(
        voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"]), model=model
    )
