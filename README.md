# The KindsOfReasoning collection of datasets

KindsOfReasoning is a collection of various datasets from various sources, testing different kinds of reasoning. 

This repository contains the original dataset and the results of various models on it, together with code to run LLMs on the dataset through the [OpenAI `evals` library](https://github.com/openai/evals).

In particular, the root folder of the repository contains the following files:
- `KindsOfReasoning.<csv|json>`: the collection of datasets in a single dataframe 
- `KindsOfReasoning_with_llm_results.<csv|json>`: the collection of datasets, with results of the tested LLMs, in a single dataframe

To use the final datasets, you can load one of those files with `pandas` or another library. If instead you want to re-create the dataset from scratch, or run it on another LLM by adapting the code used to obtain the original results, look into the `full_processing_steps` folder.

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
@misc{pacchiardi2024100instancesneedpredicting,
      title={100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances}, 
      author={Lorenzo Pacchiardi and Lucy G. Cheke and José Hernández-Orallo},
      year={2024},
      eprint={2409.03563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.03563}, 
}
```


# Credits
The collection of datasets and accompanying code is released under the [CC-BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en). This requires attribution, prohibits commercial use, and mandates that any derivatives be shared under the same terms.

## Code
Part of the code in `full_processing_steps/src/set_up_datasets.py` is adapted from https://github.com/stanford-crfm/helm, which is released under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Datasets

The individual datasets included in the `KindsOfReasoning` collection are obtained from the following sources:

- Formal Fallacies Syllogisms Negation
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Logical_Args
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Babi_Task_16
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- LogiQA 2.0
  - License: CC-BY-NC-SA 4.0
  - Attribution: https://github.com/csitfun/LogiQA2.0
- WANLI
  - License: We thank the author for having personally granted permission for the use of this dataset in this benchmark.
  - Attribution: https://github.com/alisawuffles/wanli
- Alpha_NLI
  - License: Apache License 2.0
  - Attribution: © The Allen Institute for Artificial Intelligence https://leaderboard.allenai.org/anli/submissions/get-started
- ReClor
  - License: We thank the author for having personally granted permission for the use of this dataset in this benchmark. ReClor is for non-commercial usage only
  - Attribution: https://github.com/yuweihao/reclor
- Crass_AI
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Cause and Effect
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Fantasy Reasoning
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Goal Step Inference
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Copa
  - License: BSD 2-Clause License
  - Attribution: © M Roemmele, CA Bejan, AS Gordon.  https://people.ict.usc.edu/~gordon/copa.html
- Cosmos_QA
  - License: We thank the author for having personally granted permission for the use of this dataset in this benchmark.
  - Attribution: https://github.com/wilburOne/cosmosqa
- Ropes
  - License: CC-BY 4.0
  - Attribution: © The Allen Institute for Artificial Intelligence Derived from https://huggingface.co/datasets/ropes
- ANLI
  - License: CC-BY-NC 4.0
  - Attribution: © Facebook, Inc. https://github.com/facebookresearch/anli
- Emoji_Movie
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Abstract Narrative Understanding
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Odd One Out
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Metaphor Understanding
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Geometric Shapes
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
- Space_NLI
  - License: MIT License
  - Attribution: Copyright (c) 2023 Fibo Kowalsky https://github.com/kovvalsky/SpaceNLI
- Arithmetic
  - License: Apache License 2.0
  - Attribution: Derived from [BIG-Bench](https://github.com/google/BIG-bench/).
