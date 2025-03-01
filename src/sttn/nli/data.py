import backoff
import openai
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from sttn.nli.models.output import DataProviderModel, DataProviderArgumentsModel
from sttn.nli.prompts import Context
from sttn.nli.prompts import PromptGenerator


class NetworkBuilder:
    def __init__(self, app: CompiledStateGraph, config: dict):
        self.app = app
        self.config = config

    def _ask_question(self, question: str) -> str:
        message = HumanMessage(content=question)
        stream = self.app.stream({"messages": [message]}, config=self.config)
        first_message = next(stream)
        content = first_message['model']["messages"].content
        return content

    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=2, base=8, factor=1, max_value=60)
    def pick_data_provider(self, context: Context) -> DataProviderModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderModel)
        prompt = PromptGenerator.generate_provider_prompt(context.query)
        output = self._ask_question(question=prompt)
        return parser.parse(self._sanitize_json_output(output))

    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=2, base=8, factor=1, max_value=60)
    def pick_provider_arguments(self, context: Context) -> DataProviderArgumentsModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderArgumentsModel)
        prompt = PromptGenerator.generate_data_retrieval_prompt(context)
        output = self._ask_question(question=prompt)
        return parser.parse(self._sanitize_json_output(output))

    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=2, base=8, factor=1, max_value=60)
    def get_filtering_code(self, context: Context) -> str:
        prompt = PromptGenerator.generate_data_filtering_prompt(context)
        output = self._ask_question(question=prompt)
        return output

    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=2, base=8, factor=1, max_value=60)
    def get_analysis_code(self, context: Context) -> str:
        prompt = PromptGenerator.generate_data_analysis_prompt(context)
        output = self._ask_question(question=prompt)
        return self._sanitize_output(output)

    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=2, base=8, factor=1, max_value=60)
    def get_fixed_code(self, context: Context, exc: Exception) -> str:
        exc_str = self._describe_exc(exc=exc)
        prompt = PromptGenerator.fix_analysis_code_prompt(context=context, exc_str=exc_str)
        output = self._ask_question(question=prompt)
        return self._sanitize_output(output)

    @staticmethod
    def _sanitize_output(text: str) -> str:
        _, after = text.split("```python")
        return after.split("```")[0]

    @staticmethod
    def _sanitize_json_output(text: str) -> str:
        if "```json" in text:
            _, after = text.split("```json")
            return after.split("```")[0]
        else:
            return text

    @staticmethod
    def _describe_exc(exc: Exception) -> str:
        error_type = type(exc).__name__
        error_message = str(exc)
        readable_error_string = f"{error_type}: {error_message}"
        return readable_error_string
