from typing import Optional

from IPython.core.interactiveshell import ExecutionResult
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from sttn.nli import Query
from sttn.nli.data import NetworkBuilder
from sttn.nli.prompts import Context


class STTNAnalyst:
    def __init__(self, verbose: bool = False, model_name: str = "gpt-4o", code_retry_limit: int = 1):
        self._verbose = verbose
        self._model = ChatOpenAI(temperature=0, model_name=model_name)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a chatbot having a conversation with a human."
                ),  # The persistent system prompt
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # Where the human input will injected
            ]
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._chain = LLMChain(
            llm=self._model,
            prompt=prompt,
            verbose=verbose,
            memory=memory,
        )
        self._network_builder = NetworkBuilder(model=self._chain)
        self._context: Optional[Context] = None
        self._model_name: str = model_name
        self._code_retry_limit: int = code_retry_limit

    def clarify(self, human_input: str) -> str:
        return self._chain.predict(human_input=human_input)

    def _execute_code(self, code: str) -> ExecutionResult:
        self._context.analysis_code = code
        get_ipython().set_next_input(code)
        result = get_ipython().run_cell(code)
        return result

    def _run_code_and_retry(self, code: str) -> ExecutionResult:
        result = self._execute_code(code)

        retry_counter = 0
        while result.error_in_exec is not None and retry_counter < self._code_retry_limit:
            fixed_code = self._network_builder.get_fixed_code(context=self._context, exc=result.error_in_exec)
            result = self._execute_code(fixed_code)
            print(f"attempt: {retry_counter}, code: {fixed_code}")
            retry_counter = retry_counter + 1

        return result

    def chat(self, user_query: Optional[str] = None) -> Context:
        if user_query is None:
            print("Enter your question please:")
            user_query = input()
        query = Query(user_query)
        context = Context(query=query)
        self._context = context
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
        context.data_provider_args = data_provider_args.arguments

        if not data_provider_args.feasible:
            print(f"Can not retrieve the data {args_descr}.")
            return context

        if self._verbose:
            print(args_descr)
            print(f"Data provider arguments: {data_provider_args.arguments}")

        data_provider = context.data_provider

        if self._verbose:
            print(f"Retrieving the data using {provider_id} provider with the following arguments {data_provider_args}")
        network = data_provider.get_data(**data_provider_args.arguments)
        context.network = network

        filtering_code = self._network_builder.get_filtering_code(context=context)
        # generated filtering predicates are used only as a chain-of-thought at this moment
        print(filtering_code)

        analysis_code = self._network_builder.get_analysis_code(context=context)
        content = analysis_code
        
        # add the 'context.network' to variable in user namespace to be available in InteractiveShell's (get_ipython()) scope
        get_ipython().user_ns['sttn_network'] = context.network
        result = self._run_code_and_retry(analysis_code)

        if result.error_in_exec is None:
            context.result = result.result

        return context
