# embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode(text)
        self.cache[text] = embedding
        return embedding

    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently"""
        return [self.get_embedding(text) for text in texts]