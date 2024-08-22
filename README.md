# The KindsOfReasoning collection of datasets

KindsOfReasoning is a collection of various datasets from various sources, testing different kinds of reasoning. 

This repository contains the original dataset and the results of various models on it, together with code to run LLMs on the dataset through the [OpenAI `evals` library](https://github.com/openai/evals).

In particular, the root folder of the repository contains the following files:
- `KindsOfReasoning.<csv|json>`: the collection of datasets in a single dataframe 
- `KindsOfReasoning_with_llm_results.<csv|json>`: the collection of datasets, with results of the tested LLMs, in a single dataframe

To use the final datasets, you can load one of those files with `pandas` or another library. I instead you want to re-create the dataset from 0, or run it on another LLM by adapting the code used to obtain the original results, look into the `full_processing_steps` folder.

## Models on which results are available: 

```python
        'text-ada-001',
        'text-babbage-001',
        'text-curie-001',
        'text-davinci-001',
        'text-davinci-002',
        'text-davinci-003',
        'gpt-3.5-turbo-0301',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo-0125',
        'gpt-4-0314',
        'gpt-4-0613',
        'gpt-4-1106-preview',
        'gpt-4-0125-preview',
        'gpt-4-turbo-2024-04-09',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
```

Notice that some of those models are now deprecated. You can obtain the results on other models by using our code in `full_processing_steps` or writing your own code starting from the final files (`KindsOfReasoning.<csv|json>`), which contain all prompts and system prompts.

## Citation

If you use `KindsOfReasoning`, please cite the following paper:

```bibtex


```

