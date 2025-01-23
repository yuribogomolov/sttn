import nbformat
import os
from nbconvert.preprocessors import ExecutePreprocessor
import re
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
def load_env_file():
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f".env file found at: {dotenv_path}")
        load_dotenv(dotenv_path)
    else:
        print(".env file not found.")

# Function to remove ANSI escape sequences
def remove_ansi_escape_sequences(text) -> str:
    ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', str(text))

# Function to run a Jupyter notebook with passed variables
def run_notebook(notebook_path, variables, timeout=3600) -> str:
    """
    Execute a Jupyter notebook and print the output.
    Parameters:
    - notebook_path (str): path to the notebook file.
    - variables (dict): dictionary of variables to pass to the notebook.
    - timeout (int), default 3600: timeout in seconds for each cell execution.
    
    Returns:
    - result_link (str): link to the evaluation results.

    """
    # Load the notebook content
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Notebook file '{notebook_path}' not found.")
        return ''
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return ''
    
    
    # Insert a new cell at the beginning to define the variables
    parameter_code = '\n'.join([f"{key} = {repr(value)}" for key, value in variables.items()])
    parameter_cell = nbformat.v4.new_code_cell(parameter_code)
    notebook['cells'].insert(0, parameter_cell)
    
    # Create an ExecutePreprocessor object
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    
    # Execute the notebook
    ep.preprocess(notebook, {'metadata': {'path': './'}})
    
    # Initialize the result link
    result_link = ''

    # Capture and print the output
    for cell in notebook['cells']:
        if (cell['cell_type'] == 'code'):
            for output in cell.get('outputs', []):
                if output['output_type'] == 'stream':
                    output_to_print = ''.join(output.get('text', ''))
                elif output['output_type'] == 'execute_result':
                    output_to_print = output['data'].get('text/plain', '')
                elif output['output_type'] == 'error':
                    output_to_print = 'Error: '+ ''.join(output['traceback'])
                else:
                    output_to_print = ''
                
                # Remove ANSI escape sequences from the output
                output_to_print = remove_ansi_escape_sequences(output_to_print)
                print(output_to_print)

                # Get the paragraph with link from the output
                for line in output_to_print.split('\n'):
                    if line.find('https://smith.langchain.com') != -1:
                        result_link = line
                        break
    
    return result_link

def get_env_variable(var_name, prompt_message, is_int=False, min_value=None):
    value = os.getenv(var_name)
    if value:
        print(f"{var_name} found in .env ==> {value}")
        if is_int:
            try:
                value = int(value)
                if min_value is not None and value < min_value:
                    raise ValueError
            except ValueError:
                # if the value is not an integer or less than min_value, prompt the user 
                print(f"Invalid {var_name} in .env, falling back to input.")
                value = None
    if value is None:
        while True:
            try:
                value = input(prompt_message)
                if is_int:
                    value = int(value)
                    if min_value is not None and value < min_value:
                        raise ValueError
                break
            except ValueError:
                print(f"Please enter a valid {var_name}.")
    return value

def get_api_key(var_name, prompt_message):
    if os.environ.get(var_name) is not None:
        print(f"{var_name} received!")
    else:
        print(f"Looks like your env doesn't have a {var_name}...")
        os.environ[var_name] = input(prompt_message).strip()

def main():
    print("Running an evaluation of your chain...")

    get_api_key('OPENAI_API_KEY', "Enter your OPENAI_API_KEY:")
    get_api_key('LANGCHAIN_API_KEY', "Enter your LANGCHAIN_API_KEY:")

    dataset_name = get_env_variable('DATASET_NAME',
                                    'Pick the dataset you want to evaluate your chain on (write only the number):\n1) Taxi + LEHD evaluation - 11 examples;\n2) Taxi + LEHD evaluation - 27 examples;\n3) Taxi + LEHD evaluation - 100 examples;\n4) Taxi + LEHD evaluation - 280 examples;\n5) Exit.\nNumber of the dataset:',
                                    is_int=False)
    
    model_name = get_env_variable('MODEL_NAME',
                                  'Pick the model you want to use for your chain (write only the number):\n1) GPT-4o-mini (the most cost-efficient small model);\n1) GPT-4o (the most advanced multimodal model);\n2) GPT-4 Turbo (The previous set of high-intelligence models);\n3) GPT-3.5 Turbo (Fast, inexpensive model for simple tasks);\n4) Exit.\nNumber of the model:', 
                                  is_int=False)
    
    exp_prefix = get_env_variable('EXP_PREFIX', 
                                  "Enter an experiment prefix to distinguish between different evaluations (e.g. gpt-4o-test-):")
    
    exp_version = get_env_variable('EXP_VERSION', 
                                   "Enter an experiment version to distinguish between same evaluation (eg. 1.0.0):")
    
    code_retry_limit = get_env_variable('CODE_RETRY_LIMIT', 
                                        "Enter how many times the code will be retried in case of an error (default 2):", 
                                        is_int=True, 
                                        min_value=1)
    
    max_concurrency = get_env_variable('MAX_CONCURRENCY', 
                                       "Enter the maximum number of parallel executions (WARNING: When choosing more than 1 execution the output from evaluation might get chaotic \ndon't use more than 5 executions and base it on your RAM capacity):", 
                                       is_int=True, 
                                       min_value=1)

    parameters = {
        'dataset_name': f'{dataset_name}',
        'exp_prefix': f'{exp_prefix}_',
        'exp_version': f'{exp_version}',
        'model_name': f'{model_name}',
        'code_retry_limit': code_retry_limit,
        'max_concurrency': max_concurrency,
    }
    
    notebook_path = os.path.join(os.path.dirname(__file__), 'analyst_eval.ipynb')
    print('Running the notebook...\n')
    result_link = run_notebook(notebook_path, parameters)
    
    print(f"\n------------------Link to LangSmith------------------")
    print(result_link)
    print("\n------------------You can also view the evaluation results by using the LangSmith link above------------------\n")

if __name__ == "__main__":
    load_env_file()
    main()

