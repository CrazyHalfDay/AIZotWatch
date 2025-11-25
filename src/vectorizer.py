import logging
from typing import Iterable

import numpy as np
import voyageai

logger = logging.getLogger(__name__)


class TextVectorizer:
    def __init__(
        self,
        model_name: str = "voyage-3.5",
        api_key: str = "",
        input_type: str = "document",
        batch_size: int = 128,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.input_type = input_type
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("Voyage API key is required.")
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        client = self._get_client()
        # Replace empty strings with placeholder (Voyage API rejects empty input)
        texts = [t.strip() if t and t.strip() else "[untitled]" for t in texts]
        total = len(texts)
        logger.info("Encoding %d texts with %s (batch_size=%d)", total, self.model_name, self.batch_size)

        all_embeddings = []
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug("Processing batch %d-%d of %d", i, i + len(batch), total)
            result = client.embed(
                batch,
                model=self.model_name,
                input_type=self.input_type,
            )
            all_embeddings.extend(result.embeddings)

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        # L2 normalization for FAISS IndexFlatIP (inner product = cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


__all__ = ["TextVectorizer"]
