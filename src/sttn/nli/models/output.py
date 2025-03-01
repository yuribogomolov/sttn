from typing import Dict

from pydantic import BaseModel, Field


class DataProviderModel(BaseModel):
    provider_id: str = Field(
        description="the provider id that can be used to answer the question above,"
                    " or an empty string if none of the providers from the list can be used.")
    justification: str = Field(description="reasoning for the provider selection")


class DataProviderArgumentsModel(BaseModel):
    feasible: bool = Field(
        description="True` if the data provider can retrieve data for the user query and `False` otherwise")
    justification: str = Field(description="feasibility justification")
    arguments: Dict[str, str] = Field(
        "Dictionary with data provider argument names as keys and argument values as values")


class DataAnalysisModel(BaseModel):
    analysis_code: str = Field(description="Python code that returns the answer to the user query")
