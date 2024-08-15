# Full processing steps

1. set up a dataset in a common format (script `src/set_up_datasets.py`). That downloads the raw data into the `datasets` folder and then converts them to a format suitable for the `evals` library and stores them in the `registry` folder. 
2. get LLM’s performance on that using the `evals` library (scripts in`run_evals` folder). Results are stored in `results`.



The file `1_set_up_datasets_not_used.py` includes datasets for which we originally included results but that were not used in the final collection. 

This folder contains the converted datasets (into the OpenAI evals format) for the datasets in the KindsOfReasoning collection, but it does not contain the raw files.

Question: 

- Check whether the results for all the non-included datasets are available
- Discuss why those datasets were excluded from the final collection -> simply remove them otherwise?
- upload the raw "evals" results files too? I need to do that if I keep the excluded datasets, but I do not need that if I do not keep them, as I can simply generate the final files and that would be enough?
- If I add the additional results: short script showing how to load the results for one of them and see what happens