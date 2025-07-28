# groq.py
from groq import Groq
from groq.types.chat import ChatCompletion
import os
from typing import Optional

class GroqService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def get_completion(self, prompt: str, model: str = "llama-3.3-70b-versatile", 
                     temperature: float = 0.2, max_tokens: int = 1024) -> Optional[ChatCompletion]:
        try:
            return self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Groq API error: {e}")
            return None