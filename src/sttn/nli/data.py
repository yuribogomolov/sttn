import json
from json import JSONDecodeError
from typing import Dict, Optional

import openai
from jinja2 import Environment, PackageLoader

from sttn.data.lehd import OriginDestinationEmploymentDataProvider
from sttn.data.nyc import NycTaxiDataProvider, Service311RequestsDataProvider
from sttn.network import SpatioTemporalNetwork
from sttn.nli import Query

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

    @staticmethod
    def pick_data_provider(context: Context) -> Dict[str, str]:
        prompt = NetworkBuilder._generate_provider_prompt(context.query)
        return NetworkBuilder._get_completion(prompt=prompt)

    @staticmethod
    def pick_provider_arguments(context: Context) -> Dict[str, str]:
        prompt = NetworkBuilder._generate_data_retrieval_prompt(context)
        return NetworkBuilder._get_completion(prompt=prompt)

    @staticmethod
    def get_analysis_code(context: Context) -> Dict[str, str]:
        prompt = NetworkBuilder._generate_data_analysis_prompt(context)
        return NetworkBuilder._get_completion(prompt=prompt)

    @staticmethod
    def _get_completion(prompt: str) -> Dict[str, str]:
        chat_completion = openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
        )
        content = chat_completion.choices[0].message.content
        try:
            content_js = json.loads(content)
        except JSONDecodeError as ex:
            print(f"Invalid json: {content}")
            raise ex

        return content_js

    @staticmethod
    def _generate_provider_prompt(query: Query):
        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_provider.j2")
        jcontext = {
            "user_query": query.query,
            "data_providers": DATA_PROVIDERS,
        }

        prompt = template.render(jcontext)
        return prompt

    @staticmethod
    def _generate_data_retrieval_prompt(context: Context):
        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_provider_arguments.j2")
        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_description": data_provider_instance.get_data.__doc__
        }

        prompt = template.render(jcontext)
        return prompt

    @staticmethod
    def _generate_data_analysis_prompt(context: Context):
        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_analysis.j2")
        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_description": data_provider_instance.get_data.__doc__
        }

        prompt = template.render(jcontext)
        return prompt
