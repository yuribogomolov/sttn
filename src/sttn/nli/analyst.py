from typing import Optional

from sttn.nli import Query
from sttn.nli.data import NetworkBuilder, Context


class STTNAnalyst:
    def __init__(self):
        self._verbose = True

    def chat(self, user_query: Optional[str] = None) -> Context:
        if user_query is None:
            print("Enter your question please:")
            user_query = input()
        query = Query(user_query)
        context = Context(query=query)
        data_provider = NetworkBuilder.pick_data_provider(context=context)
        provider_id = data_provider['provider_id']
        provider_descr = data_provider['justification']

        if len(provider_id) == 0:
            print(f"Don't have the data to answer the query, {provider_descr}.")
            return context

        if self._verbose:
            print(f"Picked data provider {provider_id}")
            print(provider_descr)

        context.data_provider = provider_id
        data_provider_args = NetworkBuilder.pick_provider_arguments(context)
        args_descr = data_provider_args['justification']

        if not data_provider_args['feasible']:
            print(f"Can not retrieve the data {args_descr}.")
            return context

        data_provider_args.pop('feasible', None)
        data_provider_args.pop('justification', None)
        if self._verbose:
            print(args_descr)
            print(f"Data provider arguments: {data_provider_args}")

        data_provider = context.data_provider

        if self._verbose:
            print(f"Retrieving the data using {provider_id} provider with the following arguments {data_provider_args}")
        sttn = data_provider.get_data(**data_provider_args)
        context.sttn = sttn

        analysis_code = NetworkBuilder.get_analysis_code(context=context)
        prefix = "sttn = context.sttn\n"
        analysis_code = prefix + analysis_code['analysis_code']
        get_ipython().set_next_input(analysis_code)

        return context
