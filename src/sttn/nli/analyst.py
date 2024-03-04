from typing import Optional

from langchain_openai import ChatOpenAI

from sttn.nli import Query
from sttn.nli.data import NetworkBuilder, Context


class STTNAnalyst:
    def __init__(self):
        self._verbose = True
        self._model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        self._network_builder = NetworkBuilder(model=self._model)

    def chat(self, user_query: Optional[str] = None) -> Context:
        if user_query is None:
            print("Enter your question please:")
            user_query = input()
        query = Query(user_query)
        context = Context(query=query)
        data_provider = self._network_builder.pick_data_provider(context=context)
        provider_id = data_provider.provider_id
        provider_descr = data_provider.justification

        if len(provider_id) == 0:
            print(f"Don't have the data to answer the query, {provider_descr}.")
            return context

        if self._verbose:
            print(f"Picked data provider {provider_id}")
            print(provider_descr)

        context.data_provider = provider_id
        data_provider_args = self._network_builder.pick_provider_arguments(context)
        args_descr = data_provider_args.justification

        if not data_provider_args.feasible:
            print(f"Can not retrieve the data {args_descr}.")
            return context

        if self._verbose:
            print(args_descr)
            print(f"Data provider arguments: {data_provider_args.arguments}")

        data_provider = context.data_provider

        if self._verbose:
            print(f"Retrieving the data using {provider_id} provider with the following arguments {data_provider_args}")
        sttn = data_provider.get_data(**data_provider_args.arguments)
        context.sttn = sttn

        analysis_code = self._network_builder.get_analysis_code(context=context)
        content = analysis_code.content.replace("```python\n", "").replace("```", "")
        prefix = "sttn = context.sttn\n"
        analysis_code = prefix + content
        get_ipython().set_next_input(analysis_code)

        return context
