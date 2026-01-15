import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def invoke(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content
