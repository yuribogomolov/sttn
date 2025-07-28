from openai import OpenAI
from pydantic import BaseModel

from sttn.nli import Query
from sttn.nli.prompts import Context


class ResultModel(BaseModel):
    result: float


class GPTAnalyst:
    def __init__(self, model_name: str, temperature: float = 0, verbose: bool = False):
        self.client = OpenAI()
        self.model_name = model_name
        self.verbose = verbose
        self.temperature = temperature

    def _ask_chatgpt(self, prompt: str) -> str:
        if self.verbose:
            print(f"Prompting model {self.model_name} with:\n{prompt}\n")

        system_msg = (
            "You are a helpful assistant. You respond in JSON only, "
            "with a single field called 'answer' that contains a number (integer or float), or null if unknown."
        )

        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            response_format=ResultModel,
        )
        result = response.choices[0].message.parsed
        return result

    def chat(self, user_query: str) -> Context:
        result = self._ask_chatgpt(user_query)

        query = Query(user_query)
        context = Context(query=query)
        context.result = result
        return context
