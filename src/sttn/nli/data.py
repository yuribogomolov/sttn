from sttn.nli import Query
from sttn.data.nyc import NycTaxiDataProvider, Service311RequestsDataProvider
from sttn.data.lehd import OriginDestinationEmploymentDataProvider

DATA_PROVIDERS = [NycTaxiDataProvider, Service311RequestsDataProvider, OriginDestinationEmploymentDataProvider]


class NetworkBuilder:

    @staticmethod
    def pick_data_provider(query: Query):
        prompt = NetworkBuilder._generate_provider_prompt(query)
        # call the API

    @staticmethod
    def _generate_provider_prompt(query: Query):
        prefix = f"""Pick the data provider that can be used to answer the following query.
         === Query ===
         {query.query}

         """

        data_provider_prompt = f"""=== Data Providers ===
        """
        for provider in DATA_PROVIDERS:
            data_provider_prompt = data_provider_prompt + f"""
                provider_id: {provider.__name__}
                provider_description: {provider.__doc__}
                ======
            """

        suffix = f"""===Task===
            Output a json message with two fields:
            provider_id - the provider id that can be used to answer the question above, or an empty string if none of the providers from the list can be used.
            justification - reasoning for the provider selection

        """

        return prefix + data_provider_prompt + suffix
