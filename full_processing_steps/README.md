# Complete processing steps

This folder contains all the code needed to adapt the various datasets composing `KindsOfReasoning` to the format used by the [OpenAI `evals` library](https://github.com/openai/evals), run them for the various LLMs and generate the final results files. More precisely, the complete steps are the following:

1. Download the datasets and convert them in a common format by running `Python set_up_datasets.py`. In particular, this will download the raw data into a folder named `1_raw_datasets`  and convert them to a format suitable for the [OpenAI `evals` library](https://github.com/openai/evals); the converted datasets and "registry" file are stored them in the `1_registry` folder. If you want to download one (or a few) datasets only, you can do so by running
    ```python 
    python set_up_datasets.py --datasets space_nli
    ```
2. Run `./run_evals.sh` to run the LLMs on the datasets included in KindsOfReasoning using `evals`. The results of the run will be stored in `2_run_results`. The provided version of `run_evals.sh` runs one LLM on all datasets included in `KindsOfReasoning`, by subsampling the datasets in the way done to obtain the original dataset. Before running `run_evals.sh`, you need to modify it by specifying the correct path. Moreover, running that uses the OpenAI API and thus requires specifying your OpenAI API key in `.env`. You can use the notebook `estimate_cost.ipynb` to estimate the cost of running a given LLM. 
3. Finally, generate the final dataset by running `python generate_final_dataset.py`. This will create a single dataframe with all prompts and system prompts, the success and the answers for all evaluated LLMs, and
    the various splits ("train", "validation", "test" for the different splits).

Notice that some of the datasets in `KindsOfReasoning` are already included in the `evals` library; as such, they are not downloaded in step 1 but are run in step 2.

The raw datasets (in the `1_raw_datasets` folder) and the raw results from running step 2 (in `2_run_results`) are not provided in this repository as they take up too much space. Instead, the registry `1_registry` is provided for convenience, so that users can directly jump to step 2 to evaluate a new LLM on the dataset. Notice that the `evals` library is not limited to evaluating OpenAI LLMs, see [here](https://github.com/openai/evals/blob/main/docs/completion-fns.md) for more details.



- [X] Convert to single df.
- [ ] Update cost of tokens for new models
- [ ] Run on new LLMs from OpenAI
- [ ] needed packages: Instructions on what OpenAI evals version is needed: do I need my fork, or is the default version sufficient?
