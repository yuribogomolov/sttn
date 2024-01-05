from sttn.nli import Query
from sttn.data.nyc import NycTaxiDataProvider, Service311RequestsDataProvider
from sttn.data.lehd import OriginDestinationEmploymentDataProvider

from jinja2 import Environment, PackageLoader

DATA_PROVIDERS = [NycTaxiDataProvider, Service311RequestsDataProvider, OriginDestinationEmploymentDataProvider]


class NetworkBuilder:

    @staticmethod
    def pick_data_provider(query: Query):
        prompt = NetworkBuilder._generate_provider_prompt(query)
        # call the API

    @staticmethod
    def _generate_provider_prompt(query: Query):
        environment = Environment(loader=PackageLoader("sttn", package_path="nli/templates"))
        template = environment.get_template("data_provider.j2")
        context = {
            "user_query": query.query,
            "data_providers": DATA_PROVIDERS,
        }

        prompt = template.render(context)
        return prompt
