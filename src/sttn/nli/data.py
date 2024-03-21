from typing import Optional

from jinja2 import Environment, PackageLoader
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from sttn.data.lehd import OriginDestinationEmploymentDataProvider
from sttn.data.nyc import NycTaxiDataProvider, Service311RequestsDataProvider
from sttn.network import SpatioTemporalNetwork
from sttn.nli import Query
from sttn.nli.models.output import DataProviderModel, DataProviderArgumentsModel

DATA_PROVIDERS = [NycTaxiDataProvider, Service311RequestsDataProvider, OriginDestinationEmploymentDataProvider]


class Context:
    def __init__(self, query: Query):
        self._query: Query = query

        self._data_provider_id: Optional[str] = None  # data provider id
        self._data_provider_cls = None
        self._data_provider_instance = None

        self._sttn: Optional[SpatioTemporalNetwork] = None

    @property
    def query(self):
        return self._query

    @property
    def data_provider(self):
        return self._data_provider_instance

    @data_provider.setter
    def data_provider(self, data_provider_id: str):
        self._data_provider_id = data_provider_id
        self._data_provider_cls = Context._get_data_provider_by_id(data_provider_id)
        self._data_provider_instance = self._data_provider_cls()

    @property
    def sttn(self):
        return self._sttn

    @sttn.setter
    def sttn(self, sttn: SpatioTemporalNetwork):
        self._sttn = sttn

    @staticmethod
    def _get_data_provider_by_id(data_provider_id: str):
        for data_provider in DATA_PROVIDERS:
            if data_provider.__name__ == data_provider_id:
                return data_provider

        raise KeyError(f"Can not find data provider with {data_provider_id} id.")


class NetworkBuilder:
    def __init__(self, model: LLMChain):
        self.model = model

    def pick_data_provider(self, context: Context) -> DataProviderModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderModel)
        prompt = NetworkBuilder._generate_provider_prompt(context.query)
        output = self.model.predict(human_input=prompt)
        return parser.parse(output)

    def pick_provider_arguments(self, context: Context) -> DataProviderArgumentsModel:
        parser = PydanticOutputParser(pydantic_object=DataProviderArgumentsModel)
        prompt = NetworkBuilder._generate_data_retrieval_prompt(context)
        output = self.model.predict(human_input=prompt)
        return parser.parse(output)

    def get_analysis_code(self, context: Context) -> str:
        prompt = NetworkBuilder._generate_data_analysis_prompt(context)
        output = self.model.predict(human_input=prompt)
        return self._sanitize_output(output)

    @staticmethod
    def _sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]

    @staticmethod
    def _generate_provider_prompt(query: Query) -> str:
        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_provider.j2")
        jcontext = {
            "user_query": query.query,
            "data_providers": DATA_PROVIDERS,
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def _generate_data_retrieval_prompt(context: Context) -> str:
        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_provider_arguments.j2")
        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_description": data_provider_instance.get_data.__doc__,
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def _generate_data_analysis_prompt(context: Context) -> str:
        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_analysis.j2")
        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_description": data_provider_instance.get_data.__doc__,
        }

        prompt_str = template.render(jcontext)
        return prompt_str
