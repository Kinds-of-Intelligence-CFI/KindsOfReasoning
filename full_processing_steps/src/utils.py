# list of models @ https://platform.openai.com/docs/models/continuous-model-upgrades
# cost @ https://openai.com/pricing and https://platform.openai.com/docs/deprecations/ for older models
import os

import pandas as pd

# THIS REFERS TO INPUT TOKENS AS THOSE ARE THE MAJORITY OF THE TOKENS IN OUR TESTS.
# see https://openai.com/api/pricing/
cost_per_token = {
    "gpt-4o-2024-08-06": 0.0025e-3,
    "gpt-4o-2024-05-13": 0.005e-3,
    "gpt-4o-mini-2024-07-18": 0.00015e-3,

    "gpt-4-turbo-2024-04-09": 0.01e-3,
    "gpt-4-0125-preview": 0.01e-3,
    "gpt-4-1106-preview": 0.01e-3,
    "gpt-4-0613": 0.03e-3,
    "gpt-4-32k-0613": 0.06e-3,

    "gpt-3.5-turbo-0125": 0.0005e-3,  # optimized for chat
    "gpt-3.5-turbo-1106": 0.001e-3,  # optimized for chat
    "gpt-3.5-turbo-instruct": 0.0015e-3,
    # instruct model, indicated as the replacement for old instructGPT models (similar capabilities to
    #  textx-davinci-003, compatible with legacy Completions endpoint)

    # older models, will be deprecated in June 24
    "gpt-4-0314": 0.03e-3,
    "gpt-4-32k-0314": 0.06e-3,
    "gpt-3.5-turbo-0613": 0.0015e-3,
    "gpt-3.5-turbo-16k-0613": 0.003e-3,
    "gpt-3.5-turbo-0301": 0.0015e-3,

    # even older models
    "text-ada-001": 0.0004e-3,
    "text-babbage-001": 0.0005e-3,
    "text-curie-001": 0.0020e-3,
    "text-davinci-001": 0.0200e-3,
    "text-davinci-002": 0.0200e-3,
    "text-davinci-003": 0.0200e-3,

    "ada": 0.0004e-3,
    "babbage": 0.0005e-3,
    "curie": 0.0020e-3,
    "davinci": 0.0200e-3,
}


# This dictionary is up-to-date as 2024-08-22.


def compute_n_tokens(string, tokenizer):
    return len(tokenizer(string)["input_ids"])


def compute_cost(model_name, num_tokens):
    """
    Compute the cost of the model for a given number of tokens and turns.
    """
    return cost_per_token[model_name] * num_tokens


def load_with_conditions(filename, overwrite_res=False):
    if not os.path.exists(filename) or overwrite_res:
        print("File not found or overwrite requested. Creating new dataframe.")
        df = pd.DataFrame()
    elif filename.split(".")[-1] == "csv":
        print("Loading existing dataframe.")
        df = pd.read_csv(filename)
    elif filename.split(".")[-1] == "pkl":
        print("Loading existing dataframe.")
        df = pd.read_pickle(filename)
    else:
        raise ValueError("File format not recognized. Please use .csv or .pkl.")

    return df


def save_dataframe(filename, res_df):
    if filename.endswith(".csv"):
        res_df.to_csv(filename, index=False)
    elif filename.endswith(".pkl"):
        res_df.to_pickle(filename)
    else:
        raise ValueError("filename not recognized")
