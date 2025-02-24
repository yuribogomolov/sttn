from typing import Any, Optional, Dict

import pandas as pd
from jinja2 import Environment, PackageLoader, Template

from sttn.data.brno import HealthcareDataProvider
from sttn.data.lehd import OriginDestinationEmploymentDataProvider
from sttn.data.nyc import NycTaxiDataProvider, Service311RequestsDataProvider
from sttn.network import SpatioTemporalNetwork
from sttn.nli import Query

DATA_PROVIDERS = [NycTaxiDataProvider, Service311RequestsDataProvider, OriginDestinationEmploymentDataProvider,
                  HealthcareDataProvider]


class Context:
    def __init__(self, query: Query):
        self._query: Query = query

        self._data_provider_id: Optional[str] = None  # data provider id
        self._data_provider_cls = None
        self._data_provider_instance = None

        self._data_provider_args: Optional[Dict[str, str]] = None

        self._feasible: Optional[bool] = None

        self._analysis_code: Optional[str] = None

        self._network: Optional[SpatioTemporalNetwork] = None

        self._result: Optional[Any] = None

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
    def network(self):
        return self._network

    @network.setter
    def network(self, network: SpatioTemporalNetwork):
        self._network = network

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result: Any):
        self._result = result

    @property
    def data_provider_id(self):
        return self._data_provider_id

    @data_provider_id.setter
    def data_provider_id(self, data_provider_id: str):
        self._data_provider_id = data_provider_id

    @property
    def data_provider_args(self):
        return self._data_provider_args

    @data_provider_args.setter
    def data_provider_args(self, _data_provider_args: Dict[str, str]):
        self._data_provider_args = _data_provider_args

    @property
    def feasible(self):
        return self._feasible

    @feasible.setter
    def feasible(self, feasible: bool):
        self._feasible = feasible

    @property
    def analysis_code(self):
        return self._analysis_code

    @analysis_code.setter
    def analysis_code(self, analysis_code: str):
        self._analysis_code = analysis_code

    @staticmethod
    def _get_data_provider_by_id(data_provider_id: str):
        for data_provider in DATA_PROVIDERS:
            if data_provider.__name__ == data_provider_id:
                return data_provider

        raise KeyError(f"Can not find data provider with {data_provider_id} id.")


class PromptGenerator:

    @staticmethod
    def get_df_description(df: pd.DataFrame) -> str:
        schema_str = ""
        for col, dtype in df.dtypes.items():
            if dtype.name == 'geometry':
                schema_str += f"{col}: geometry\n"
            elif pd.api.types.is_numeric_dtype(dtype):
                # Calculate min, max, and mean for numeric columns
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                has_negative = min_val < 0
                schema_str += f"{col:20}: {dtype} - Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f}, Has negative values: {has_negative}\n"
            else:
                # Find the 10 most common values for string columns
                common_vals = df[col].value_counts().head(10)
                common_vals_str = ", ".join([f"{val} ({count})" for val, count in common_vals.items()])
                schema_str += f"{col:20}: {dtype} - Most common values: {common_vals_str}\n"
        return schema_str

    @staticmethod
    def get_template(template_fname: str) -> Template:
        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        return environment.get_template(template_fname)

    @staticmethod
    def generate_provider_prompt(query: Query) -> str:
        template = PromptGenerator.get_template("data_provider.j2")
        jcontext = {
            "user_query": query.query,
            "data_providers": DATA_PROVIDERS,
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def generate_data_retrieval_prompt(context: Context) -> str:
        template = PromptGenerator.get_template("data_provider_arguments.j2")

        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_description": data_provider_instance.get_data.__doc__,
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def generate_data_filtering_prompt(context: Context) -> str:
        template = PromptGenerator.get_template("data_filter.j2")

        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider
        nodes_description = PromptGenerator.get_df_description(context.network.nodes)
        edges_description = PromptGenerator.get_df_description(context.network.edges)

        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_provider_arguments": context.data_provider_args,
            "data_description": data_provider_instance.get_data.__doc__,
            "nodes_description": nodes_description,
            "edges_description": edges_description
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def generate_data_analysis_prompt(context: Context) -> str:
        template = PromptGenerator.get_template("data_provider_arguments.j2")

        data_provider = context.data_provider.__class__
        data_provider_instance = context.data_provider

        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_analysis.j2")
        jcontext = {
            "user_query": context.query.query,
            "data_provider_documentation": data_provider.__doc__,
            "data_provider_arguments": context.data_provider_args,
            "data_description": data_provider_instance.get_data.__doc__,
        }

        prompt_str = template.render(jcontext)
        return prompt_str

    @staticmethod
    def fix_analysis_code_prompt(context: Context, exc_str: str) -> str:
        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("code_execution_error.j2")
        jcontext = {
            "user_query": context.query.query,
            "executed_code": context.analysis_code,
            "error": exc_str,
        }

        prompt_str = template.render(jcontext)
        return prompt_str
