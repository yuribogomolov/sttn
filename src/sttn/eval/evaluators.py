from langsmith.schemas import Example, Run
from langsmith.evaluation import LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langsmith import traceable
import openai
import backoff
from typing import List

# Parameters for exponential backoff
max_tries=6  # max tries before giving up
base=8  # initial backoff time in seconds
factor=1  # backoff factor
max_value=60  # max backoff time in seconds


# reformat the whole file as class
class Evaluators:
    def __init__(self):
        # Evaluating LLM
        self.eval_llm = ChatOpenAI(temperature=0.0, model="gpt-4o", max_retries=3)
    ######--------------------------------------- EVALUATORS (evaluate each example) ---------------------------------------######

    def data_provider_id_match(self, run: Run, example: Example) -> dict:
        ref_provider_id = example.outputs["data_provider_id"]
        pred_provider_id = run.outputs["data_provider_id"]
        score = pred_provider_id == ref_provider_id
        return {"key": "data_provider_match",
                "score": int(score)}

    def data_provider_args_match(self, run: Run, example: Example) -> dict:
        ref_provider_args = example.outputs["data_provider_args"]
        pred_provider_args = run.outputs["data_provider_args"]
        score = pred_provider_args == ref_provider_args
        return {"key": "data_provider_args_match",
                "score": int(score)}

    def result_match(self, run: Run, example: Example) -> dict:
        try:
            if example.outputs["result"] in [None, "", "null", "Null", "NULL", "None", "none"]:
                ref_result = None
            else:
                ref_result = example.outputs["result"] = float(example.outputs["result"])
                ref_result = round(ref_result, 5)
            
            if run.outputs["result"] in [None, "", "null", "Null", "NULL", "None", "none"]:
                pred_result = None
            else:
                pred_result = run.outputs["result"] = float(run.outputs["result"])
                pred_result = round(pred_result, 5)

            pred_result = run.outputs["result"]
            score = pred_result == ref_result

            return {"key": "result_match",
                    "score": int(score)}
        except KeyError as e:
            print(f"KeyError: `{str(e)}` attribute is MISSING in the dataset\n")
            return {"key": "result_match",
                    "score": -1}
        except Exception as e:
            print(f"An error happened while using `result_match`: {str(e)}\n")
            return {"key": "result_match",
                    "score": 0}

    def executable_match(self, run: Run, example: Example) -> dict:
        ref_provider_args = example.outputs["executable"]
        pred_provider_args = run.outputs["executable"]
        score = pred_provider_args == ref_provider_args
        return {"key": "executable",
                "score": bool(score)}

    ######--------------------------------------- LLM EVALUATORS (evaluate each example) ---------------------------------------######
    
    def _get_geosp_aware_eval(self):
        geosp_aware_eval = LangChainStringEvaluator(
            "criteria",# "labeled_score_string", 
            config={
                # Use eval_llm to evaluate the code 
                "llm": self.eval_llm,
                "criteria": {            
                    # Correct naming (e.g. Manhattan, Staten Island counties doesn't exist (only boroughs), it's New York and Richmond counties), identification (e.g. didn't pick a street or city with similar/same name instead of requested county or district) and use of geographic entities (e.g., counties, cities, census tracts, taxi zones, zip codes, districts). 
                    "geospatial_awareness_llm_eval": "Evaluate whether the assistanse AI properly accounted for the geospatial features and relationships in the code based on the received SpatioTemporalNetwork and user's input query.\nThe evaluation should consider: \
                                            \n1.Abscence of naming overlap (e.g., didn't pick wrong entity (e.g., street or city with similar/same name) instead of requested entity (e.g., county or district)).\
                                            \n2.Proper interchangeability and unit use (if our data provider has only official administrative units (e.g., counties) and query asks for the same interchangable entity (e.g., borough) the model should pick the right entity that is available in data provider).\
                                            \n3.Accurate relational understanding of hierarchical and nested geographic entities (e.g., which counties are located within a city, the relationship between different geographic levels). \
                                            \n4.Proper filtering and aggregation based on the geographic criteria IF specified in the input query (e.g. uses all corresponding and available counties, census tracts, districts, zones to represent a bigger entity like state, city, etc.). \
                                            \n5.Ability to handle various regional features specific to different countries (e.g., \"okresy\" in Czechia, \"departments\" in France). \
                                            \n6.Accurate spatial relationships such as identifying entities within a certain distance, to any cardinal direction, or from/in between other geographic entities (if asked in query). \
                                            \nAnd any other geospatial peculiarities that are relevant to the input query. \
                                            \nProvide the score ONLY for addressing the main geospatial requirements of the input query,\
                                            ignoring all other errors or inconsistencies unrelated to the geospatial feature handling in the code (like code efficiency, columns/index mismatches, previous errors, etc.)\
                                            If the query doesn't require handling geospatial features, provide the score of 1.",
                    },
            },
            prepare_data=lambda run, example: {
                "prediction": run.outputs["analysis_code"], 
                "input": example.inputs["question"],
            }  
        )
        return geosp_aware_eval
        
    def _get_temp_aware_eval(self):
        temp_aware_eval = LangChainStringEvaluator(
            # 1.Assess if the code can handle temporal features specific to different contexts or regions (e.g., fiscal years in different countries, cultural calendars, different public holidays).\
            # 2.Check if the code accurately filters and aggregates  temporal features and ensure that all relevant temporal units are considered for for it (e.g., summing up daily data to get monthly totals)
            # 3.Ensure that there are no temporal inconsistencies (e.g., overlapping time periods, mismatched time zones, not acconting for leap years).\
            # 4.Ensure that temporal calculations (e.g., moving averages, time intervals, days/hours between two dates) are precise and correct (if used).\
            "criteria",
            config={
                # Use eval_llm to evaluate the code 
                "llm": self.eval_llm,
                "criteria": {
                    "temporal_awareness_llm_eval": "Evaluate whether the assistant AI properly accounted for the temporal features and relationships in the code based on the received SpatioTemporalNetwork and user's input query.\nThe evaluation should consider: \
                                        \n1.Accurate use of temporal features specific to different contexts or regions (e.g., different public holidays, cultural calendars, fiscal years in different countries).\
                                        \n2.Proper filtering and aggregation of temporal features, ensuring that all relevant temporal units are considered for it (e.g., summing up daily data to get monthly totals).\
                                        \n3.Absence of temporal inconsistencies (e.g., overlapping time periods, mismatched time zones, not accounting for leap years).\
                                        \n4.Temporal calculations (e.g., moving averages, time intervals, days/hours between two dates) are properly done (if used).\
                                        \nAnd any other temporal peculiarities that are relevant to the input query. \
                                        \nProvide the score ONLY for addressing the main temporal requirements of the input query,\
                                            ignoring all other errors or inconsistencies unrelated to the temporal feature handling in the code (like code efficiency, columns/index mismatches, previous errors, etc.)\
                                            If the query doesn't require handling temporal features, provide the score of 1.0.",
                },
            },
            prepare_data=lambda run, example: {
                "prediction": run.outputs["analysis_code"], 
                "input": example.inputs["question"],
            }
        )
        return temp_aware_eval

    # Create geospatial awareness evaluator function with backoff
    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=max_tries, base=base, factor=factor, max_value=max_value)
    @traceable
    def get_geosp_aware_eval_score(self, run: Run, example: Example) -> float:
        try:
            # Evaluate the geospatial awareness
            if "geospatial awareness" in example.outputs['categories']:
                result = self._get_geosp_aware_eval().as_run_evaluator()(run, example)
                # Return the score and evaluating LLM output text
                if result.score in [None, ""]:
                    score = 0.5
                else:
                    score = result.score
                return {"key": "geospatial_awareness_llm",
                        "score": score}
            else:
                return {"key": "__ignore",
                        "score": -1.0}
        except Exception as e:
            print(f"An error occurred while evaluating geospatial awareness with LLM: {str(e)}\n")

    # Create temporal awareness evaluator function with backoff
    @backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=max_tries, base=base, factor=factor, max_value=max_value)
    @traceable
    def get_temp_aware_eval_score(self, run: Run, example: Example) -> float:
        try:
            if "temporal awareness" in example.outputs['categories']:
                # Evaluate the temporal awareness
                result = self._get_temp_aware_eval().as_run_evaluator()(run,example)
                # Return the score and evaluating LLM output text
                if result.score in [None, ""]:
                    score = 0.5
                else:
                    score = result.score
                return {"key": "temporal_awareness_llm",
                        "score": score}#, result.value
            else:
                return {"key": "__ignore",
                        "score": -1.0}
        except Exception as e:
            print(f"An error occurred while evaluating temporal awareness with LLM: {str(e)}\n")
    


######---------------------------------- SUMMARY EVALUATORS (evaluate all examples) ----------------------------------######

class SummaryEvaluators:
    def __init__(self):
        self.evaluators = Evaluators()
    #--------------------------------------- Specific data_provider_`ids` ---------------------------------------#
    # Taxi
    def taxi_dp_id_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for accuracy score only for NycTaxiDataProvider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "taxi_dp_id_accuracy",
                 "score": float} - the accuracy score of the correct matches of NycTaxiDataProvider
        """
        try:
            sum_id_match = 0
            taxi_examples = 0
            for run, example in zip(runs, examples):
                # Check if NycTaxiDataProvider used
                if example.outputs["data_provider_id"] == "NycTaxiDataProvider":
                    sum_id_match += self.evaluators.data_provider_id_match(run, example)['score']
                    taxi_examples += 1
            
            return {"key": "taxi_dp_id_accuracy",
                    "score": sum_id_match/taxi_examples}
        except Exception as e:
            print(f"An error occurred while using NycTaxiDataProvider accuracy summary evaluator: {str(e)}")
            print("`NycTaxiDataProvider` might be MISSING in the dataset\n")
            return {"key": "taxi_dp_id_accuracy",
                    "score": -1.0}

    # LEHD
    def lehd_dp_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for id accuracy only for OriginDestinationEmploymentDataProvider data provider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "lehd_dp_id_accuracy",
                "score": float} - the accuracy score of the correct matches of OriginDestinationEmploymentDataProvider
        """
        try:
            sum_id_match = 0
            lehd_examples = 0
            for run, example in zip(runs, examples):
                # Check if OriginDestinationEmploymentDataProvider used
                if example.outputs["data_provider_id"] == "OriginDestinationEmploymentDataProvider":
                    sum_id_match += self.evaluators.data_provider_id_match(run, example)['score']
                    lehd_examples += 1
            
            return {"key": "lehd_dp_id_accuracy",
                    "score": sum_id_match/lehd_examples}
        
        except Exception as e:
            print(f"An error occurred while using `LEHD data provider` accuracy summary evaluator: {str(e)}")
            print("`OriginDestinationEmploymentDataProvider` might be MISSING in the dataset\n")
            return {"key": "lehd_dp_id_accuracy",
                    "score": -1.0}

    #--------------------------------------- Specific data_provider_`args` (when `id` predicted correctly) ---------------------------------------#
    # Taxi
    def taxi_dp_args_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for args accuracy for NycTaxiDataProvider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "taxi_dp_args_accuracy",
                "score": float} - the accuracy score for the matched args of NycTaxiDataProvider
        """
        try:
            taxi_examples = 0
            sum_args_match = 0
            for run, example in zip(runs, examples):
                # Check if NycTaxiDataProvider used
                if example.outputs["data_provider_id"] == "NycTaxiDataProvider":
                    taxi_examples += 1
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']

                    # check the args only when id was predicted correctly            
                    if id_match:
                        args_match = self.evaluators.data_provider_args_match(run, example)['score']
                        sum_args_match += args_match
        
            return {"key": "taxi_dp_args_accuracy",
                    "score": sum_args_match/taxi_examples}
        except Exception as e:
            print(f"An error occurred while using `NycTaxiDataProvider` 'args' accuracy summary evaluator: {str(e)}")
            print("`NycTaxiDataProvider` might be MISSING in the dataset\n")
            return {"key": "taxi_dp_args_accuracy",
                    "score": -1.0}

    # LEHD
    def lehd_dp_args_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """ 
        Evaluator for args accuracy for OriginDestinationEmploymentDataProvider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "lehd_dp_args_accuracy",
                "score": float} - the accuracy score for the matched args of OriginDestinationEmploymentDataProvider
        """
        try:
            lehd_examples = 0
            sum_args_match = 0
            for run, example in zip(runs, examples):
                # Check if OriginDestinationEmploymentDataProvider used
                if example.outputs["data_provider_id"] == "OriginDestinationEmploymentDataProvider":
                    lehd_examples += 1
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    
                    # check the args only when id was predicted correctly
                    if id_match:
                        args_match = self.evaluators.data_provider_args_match(run, example)['score']
                        sum_args_match += args_match
            
            return {"key": "lehd_dp_args_accuracy",
                    "score": sum_args_match/lehd_examples}
        except Exception as e:
            print(f"An error occurred while using `LEHD data provider` 'args' accuracy summary evaluator: {str(e)}")
            print("`OriginDestinationEmploymentDataProvider` might be MISSING in the dataset\n")
            return {"key": "lehd_dp_args_accuracy",
                    "score": -1.0}
        
    #--------------------------------------- Specific data_provider `result` (when `id` and `args` predicted correctly) ---------------------------------------#
    # Taxi
    def taxi_dp_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for NycTaxiDataProvider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "taxi_dp_result_accuracy",
                "score": float} - the accuracy score for the 'result' of NycTaxiDataProvider examples
        """
        try:
            sum_result_match = 0
            taxi_examples = 0
            for run, example in zip(runs, examples):
                if example.outputs["data_provider_id"] == "NycTaxiDataProvider":
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    taxi_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']
                        
                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "taxi_dp_result_accuracy",
                    "score": sum_result_match/taxi_examples}
        
        except Exception as e:
            print(f"An error occurred while using `NycTaxiDataProvider` 'result' accuracy summary evaluator: {str(e)}")
            print("`NycTaxiDataProvider` or 'result' attribute might be MISSING in the dataset\n")
            return {"key": "taxi_dp_result_accuracy",
                    "score": -1.0}
        
    # LEHD
    def lehd_dp_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for OriginDestinationEmploymentDataProvider.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "lehd_dp_result_accuracy",
                "score": float} - the accuracy score for the 'result' of OriginDestinationEmploymentDataProvider examples
        """
        try:
            sum_result_match = 0
            lehd_examples = 0
            for run, example in zip(runs, examples):
                if example.outputs["data_provider_id"] == "OriginDestinationEmploymentDataProvider":
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    lehd_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "lehd_dp_result_accuracy",
                    "score": sum_result_match/lehd_examples}
        
        except Exception as e:
            print(f"An error occurred while using `LEHD data provider` 'result' accuracy summary evaluator: {str(e)}")
            print("`OriginDestinationEmploymentDataProvider` or 'result' attribute might be MISSING in the dataset\n")
            return {"key": "lehd_dp_result_accuracy",
                    "score": -1.0}
    ###

    ###--------------------------------------- Categories ---------------------------------------###
    ###
    #--------------------------------------- Geo-spatial awareness ---------------------------------------#
    def geospatial_awr_llm_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for geospatial awareness accuracy.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "geospat_awr_llm_eval",
                "score": float} - the accuracy score for the geospatial awareness of the examples
        """

        try:
            sum_geospatial_awr = 0
            geospatial_examples = 0
            for run, example in zip(runs, examples):
                if "geospatial awareness" in example.outputs['categories']:
                    geospatial_examples += 1
                    geosp_aware_score = run.feedback_stats['geospatial_awareness']['avg'] # geosp_aware_score, geosp_aware_output
                    if geosp_aware_score != None:
                        sum_geospatial_awr += geosp_aware_score
                    # elif geosp_aware_output != None:
                    #     sum_geospatial_awr += 
            
            return {"key": "geospat_awr_llm_eval",
                    "score": sum_geospatial_awr/geospatial_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating geospatial awareness with LLM: {str(e)}")
            print("`geospatial awareness` attribute might be MISSING in the dataset\n")
            return {"key": "geospat_awr_llm_eval",
                    "score": -1.0}

    def geospatial_awr_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that require geospatial awareness.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "geospat_awr_result_accuracy",
                "score": float} - the accuracy score for the 'result' of geospatial awareness examples
        """
        try:
            sum_result_match = 0
            geospatial_examples = 0
            for run, example in zip(runs, examples):
                if "geospatial awareness" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    geospatial_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "geospat_awr_result_accuracy",
                    "score": sum_result_match/geospatial_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating `results` accuracy of examples that require `geospatial awareness`: {str(e)}")
            print("`geospatial awareness` or 'result' attributes might be MISSING in the dataset\n")
            return {"key": "geospat_awr_result_accuracy",
                    "score": -1.0}

    # --------------------------------------- Temporal awareness ---------------------------------------#
    def temporal_awr_llm_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for temporal awareness accuracy.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "temp_awr_llm_eval",
                "score": float} - the accuracy score for the temporal awareness of the examples
        """
        try:
            sum_temporal_awr = 0
            temporal_examples = 0
            for run, example in zip(runs, examples):
                if "temporal awareness" in example.outputs['categories']:
                    temporal_examples += 1
                    temp_aware_score = run.feedback_stats['temporal_awareness']['avg']
                    if temp_aware_score != None:
                        sum_temporal_awr += temp_aware_score

            return {"key": "temp_awr_llm_eval",
                    "score": sum_temporal_awr/temporal_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating temporal awareness with LLM: {str(e)}")
            print("`temporal awareness` attribute might be MISSING in the dataset\n")
            return {"key": "temp_awr_llm_eval",
                    "score": -1.0}

    def temporal_awr_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that require temporal awareness.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "temp_awr_result_accuracy",
                "score": float} - the accuracy score for the 'result' of temporal awareness examples
        """
        try:
            sum_result_match = 0
            temporal_examples = 0
            for run, example in zip(runs, examples):
                if "temporal awareness" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    temporal_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "temp_awr_result_accuracy",
                    "score": sum_result_match/temporal_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that require `temporal awareness`: {str(e)}")
            print("`temporal awareness` or `result` attributes might be MISSING in the dataset\n")
            return {"key": "temp_awr_result_accuracy",
                    "score": -1.0}

    # --------------------------------------- Community detection ---------------------------------------#
    def comm_det_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that use community detection.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "comm_det_result_accuracy",
                "score": float} - the accuracy score for the 'result' of community detection examples
        """
        try:
            sum_result_match = 0
            comm_det_examples = 0
            for run, example in zip(runs, examples):
                if "community detection" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    comm_det_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "comm_det_result_accuracy",
                    "score": sum_result_match/comm_det_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that use `community detection`: {str(e)}")
            print("`community detection` or `result` attributes might be MISSING in the dataset\n")
            return {"key": "comm_det_result_accuracy",
                    "score": -1.0}
    # --------------------------------------- PageRank ---------------------------------------#
    def pagerank_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that use PageRank.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "pagerank_result_accuracy",
                "score": float} - the accuracy score for the 'result' of PageRank examples
        """
        try:
            sum_result_match = 0
            pagerank_examples = 0
            for run, example in zip(runs, examples):
                if "pagerank" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    pagerank_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "pagerank_result_accuracy",
                    "score": sum_result_match/pagerank_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that use `PageRank`: {str(e)}")
            print("`PageRank` or `result` attributes might be MISSING in the dataset\n")
            return {"key": "pagerank_result_accuracy",
                    "score": -1.0}
    # --------------------------------------- Network Density ---------------------------------------#
    def net_dens_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that use network density.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "net_dens_result_accuracy",
                "score": float} - the accuracy score for the 'result' of network density examples
        """
        try:
            sum_result_match = 0
            net_dens_examples = 0
            for run, example in zip(runs, examples):
                if "network density" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    net_dens_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "net_dens_result_accuracy",
                    "score": sum_result_match/net_dens_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that use `network density`: {str(e)}")
            print("`network density` or `result` attributes might be MISSING in the dataset\n")
            return {"key": "net_dens_result_accuracy",
                    "score": -1.0}
    # --------------------------------------- Degree Centrality ---------------------------------------#
    def cen_deg_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that use degree centrality.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "cen_deg_result_accuracy",
                "score": float} - the accuracy score for the 'result' of degree centrality examples
        """
        try:
            sum_result_match = 0
            cen_deg_examples = 0
            for run, example in zip(runs, examples):
                if "centrality degree" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    cen_deg_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "cen_deg_result_accuracy",
                    "score": sum_result_match/cen_deg_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that use `centrality degree`: {str(e)}")
            print("`centrality degree` or `result` attributes might be MISSING in the dataset\n")
            return {"key": "cen_deg_result_accuracy",
                    "score": -1.0}
    # --------------------------------------- Clustering Coefficient ---------------------------------------#
    def clust_coef_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Evaluator for 'result' accuracy for examples that use clustering coefficient.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "clust_coef_result_accuracy",
                "score": float} - the accuracy score for the 'result' of clustering coefficient examples
        """
        try:
            sum_result_match = 0
            clust_coef_examples = 0
            for run, example in zip(runs, examples):
                if "clustering coefficient" in example.outputs["categories"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    clust_coef_examples += 1

                    # check the args only when id was predicted correctly
                    if id_match:
                        id_args_match = self.evaluators.data_provider_args_match(run, example)['score']

                        # check the result only when id and args were predicted correctly
                        if id_args_match:
                            sum_result_match += self.evaluators.result_match(run, example)['score']
            
            return {"key": "clust_coef_result_accuracy",
                    "score": sum_result_match/clust_coef_examples}
        
        except Exception as e:
            print(f"An error occurred while evaluating results accuracy of examples that use `clustering coefficient`: {str(e)}")
            print("`clustering coefficient` or `result` attributes might be MISSING IN the dataset\n")
            return {"key": "clust_coef_result_accuracy",
                    "score": -1.0}

    # --------------------------------------- Poorly-written ---------------------------------------#
    def poorly_written_args_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Accuracy evaluator for the poorly written queries.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "poorly_written_args_accuracy",
                "score": float} - the accuracy score for the poorly written queries
        """
        try:
            poorly_written_correct_id_and_args = 0
            poorly_written_examples = 0

            for run, example in zip(runs, examples):
                if example.outputs["poorly_written"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    poorly_written_examples += 1
                    if id_match:
                        args_match = self.evaluators.data_provider_args_match(run, example)['score']
                        poorly_written_correct_id_and_args += args_match
            
            return {"key": "poorly_written_args_accuracy",
                    "score": poorly_written_correct_id_and_args/poorly_written_examples}
        
        except KeyError as e:
            print(f"KeyError: {str(e)} attribute is missing in the dataset")
            return {"key": "poorly_written_args_accuracy",
                    "score": -1.0}
        except Exception as e:
            print(f"An error occurred while using `poorly-written args` accuracy summary evaluator: {str(e)}\n")
            return {"key": "poorly_written_args_accuracy",
                    "score": -1.0}

    def poorly_written_result_accuracy_summary_eval(self, runs: List[Run], examples: List[Example]) -> dict:
        """
        Accuracy evaluator for the poorly written queries.
        ### Parameters:
        - runs: List[Run] - list of LangChain Run objects
        - examples: List[Example] - list of Langsmith Example objects (from the test dataset)

        ### Returns:
        - dict: {"key": "poorly_written_result_accuracy",
                "score": float} - the accuracy score for the poorly written queries
        """
        try:
            poorly_written_correct_id_and_args = 0
            poorly_written_examples = 0

            for run, example in zip(runs, examples):
                if example.outputs["poorly_written"]:
                    id_match = self.evaluators.data_provider_id_match(run, example)['score']
                    poorly_written_examples += 1
                    if id_match:
                        args_match = self.evaluators.data_provider_args_match(run, example)['score']
                        if args_match:
                            poorly_written_correct_id_and_args += self.evaluators.result_match(run, example)['score']
            
            return {"key": "poorly_written_result_accuracy",
                    "score": poorly_written_correct_id_and_args/poorly_written_examples}
        
        except KeyError as e:
            print(f"KeyError: {str(e)} attribute is MISSING in the dataset")
            return {"key": "poorly_written_result_accuracy",
                    "score": -1.0}
        except Exception as e:
            print(f"An error occurred while using `poorly-written results` accuracy summary evaluator: {str(e)}")
            print("`poorly_written` feature might be MISSING in the dataset")
            return {"key": "poorly_written_result_accuracy",
                    "score": -1.0}

