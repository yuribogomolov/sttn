from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from sttn.nli.prompts import Context
from sttn.nli.prompts import PromptGenerator
from sttn.nli.models.output import DataProviderModel, DataProviderArgumentsModel


class NetworkBuilder:
    def __init__(self, model: LLMChain):
        self.model = model

    def pick_data_provider(self, context: Context) -> DataProviderModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderModel)
        prompt = PromptGenerator.generate_provider_prompt(context.query)
        output = self.model.predict(human_input=prompt)
        return parser.parse(output)

    def pick_provider_arguments(self, context: Context) -> DataProviderArgumentsModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderArgumentsModel)
        prompt = PromptGenerator.generate_data_retrieval_prompt(context)
        output = self.model.predict(human_input=prompt)
        return parser.parse(output)

    def get_filtering_code(self, context: Context) -> str:
        prompt = PromptGenerator.generate_data_filtering_prompt(context)
        output = self.model.predict(human_input=prompt)
        return output

    def get_analysis_code(self, context: Context) -> str:
        prompt = PromptGenerator.generate_data_analysis_prompt(context)
        output = self.model.predict(human_input=prompt)
        return self._sanitize_output(output)

    def get_fixed_code(self, context: Context, exc: Exception) -> str:
        exc_str = self._describe_exc(exc=exc)
        prompt = PromptGenerator.fix_analysis_code_prompt(context=context, exc_str=exc_str)
        output = self.model.predict(human_input=prompt)
        return self._sanitize_output(output)

    @staticmethod
    def _sanitize_output(text: str) -> str:
        _, after = text.split("```python")
        return after.split("```")[0]

    @staticmethod
    def _describe_exc(exc: Exception) -> str:
        error_type = type(exc).__name__
        error_message = str(exc)
        readable_error_string = f"{error_type}: {error_message}"
        return readable_error_string
