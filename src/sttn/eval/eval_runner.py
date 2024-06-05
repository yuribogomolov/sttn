import nbformat
import os
from nbconvert.preprocessors import ExecutePreprocessor
import re


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
        if cell['cell_type'] == 'code':
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
                    


if __name__ == "__main__":
    print("Running an evaluation of your chain...")

    # Check if the env variable OPENAI_API_KEY exists
    if os.environ.get('OPENAI_API_KEY') is not None:
        print("OPENAI_API_KEY received!")
    else:
        print("Looks like your env doesn't have an OPENAI_API_KEY...")
        os.environ['OPENAI_API_KEY'] = input("Enter your OPENAI_API_KEY:").strip()
    
    # Check if the env variable LANGCHAIN_API_KEY exists
    if os.environ.get('LANGCHAIN_API_KEY') is not None:
        print("LANGCHAIN_API_KEY received!")
    else:
        print("Looks like your env doesn't have a LANGCHAIN_API_KEY...")
        os.environ['LANGCHAIN_API_KEY'] = input("Enter your LANGCHAIN_API_KEY:").strip()

    # Interactive dataset load menu for chain evaluation
    while True:
        print('Pick the dataset you want to evaluate your chain on (write only the number):')
        print('1) Taxi provider eval (5 experiments, id-arg-exec check);')
        print('2) Taxi+LEHD providers eval (25 experiments, id-arg-exec check);')
        #print('3) All providers eval (100 experiments, id-arg-exec-result check);')
        print('4) Exit.')
        picked_dataset_num = input("Number of the dataset:")
        if picked_dataset_num.find('1') != -1:
            dataset_name = 'Taxi provider eval small1'
            break
        elif picked_dataset_num.find('2') != -1:
            dataset_name = 'Taxi + lehd evaluation - 25 examples'
            break
        elif picked_dataset_num.find('3') != -1:
            #dataset_name = 'All providers eval'
            #break
            pass
        elif picked_dataset_num.find('4') != -1:
            print("Exiting...")
            exit()
        else:
            print("Invalid dataset number, try again!")
    
    # Experiment prefix to distinguish between different evaluations
    exp_prefix = input("Enter an experiment prefix to distinguish between different evaluations:")
    exp_version = input("Enter an experiment version to distinguish between same evaluation (eg. 1.0.0):")

    # Interactive model load menu for chain evaluation
    while True:
        print('Pick the model you want to use for your chain (write only the number):')
        print('1) GPT-4o (The fastest and most affordable flagship model);')
        print('2) GPT-4 Turbo (The previous set of high-intelligence models);')
        print('3) GPT-3.5 Turbo (Fast, inexpensive model for simple tasks);')
        print('4) Exit.')
        picked_dataset_num = input("Number of the model:")
        if picked_dataset_num.find('1') != -1:
            model_name = 'gpt-4o'
            break
        elif picked_dataset_num.find('2') != -1:
            model_name = 'gpt-4-turbo'
            break
        elif picked_dataset_num.find('3') != -1:
            model_name = 'gpt-3.5-turbo'
            break
        elif picked_dataset_num.find('4') != -1:
            print("Exiting...")
            exit()
        else:
            print("Invalid model number, try again!")

    # Define the variables you want to pass
    parameters = {
        'dataset_name': f'{dataset_name}',
        'exp_prefix': f'{exp_prefix}_',
        'exp_version': f'{exp_version}',
        'model_name': f'{model_name}'
    }
    
    notebook_path = 'analyst_eval.ipynb'
    result_link = run_notebook(notebook_path, parameters)
    print(result_link)
    print("------------------You can also view the evaluation results by using the LangSmith link above------------------")


