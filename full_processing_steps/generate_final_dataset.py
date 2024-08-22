import os
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

sort_models_order = ['ada',
                     'text-ada-001',
                     'babbage',
                     'text-babbage-001',
                     'curie',
                     'text-curie-001',
                     'davinci',
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
                     ]


class ResultsLoader(ABC):
    default_llms: List[str]

    def __init__(self, llms=None, base_path_raw=None, processed_filename=None, verbose=False, load_results_kwargs={},
                 load_processed_kwargs={}):
        self.verbose = verbose

        if llms is None:
            self.llms = self.default_llms
        else:
            self.llms = llms

        if processed_filename is not None:
            self.results_df = self.load_processed(processed_filename, **load_processed_kwargs)
            # discard all rows for which no success has been recorded for any llm
            self.results_df = self.results_df[
                ~self.results_df[[f"Success_{llm}" for llm in self.llms]].isna().all(axis=1)]
        else:
            if not base_path_raw is None:
                # add it to the kwargs
                load_results_kwargs["base_path"] = base_path_raw

            self.results_df = self.load_results(**load_results_kwargs)
            # shuffle the dataframe
            self.results_df = self.results_df.sample(frac=1, random_state=42)
            # reset index
            self.results_df = self.results_df.reset_index(drop=True)

    @abstractmethod
    def load_results(self, base_path):
        """This should return a single dataframe which includes the prompt, a column for each feature and a success
         column for each llm"""
        pass

    @staticmethod
    def load_processed(filename, compression=False, compression_kwargs=None):
        """
        Load the data from the processed json file and return a DataFrame with the data.
        """
        if not compression:
            compression_kwargs = None
        elif compression and compression_kwargs is None:
            compression_kwargs = {'method': 'gzip', 'compresslevel': 1, 'mtime': 1}

        return pd.read_json(filename, orient="columns", compression=compression_kwargs)

    def save_processed(self, filename, compression=False, compression_kwargs=None):
        """
        Save the processed DataFrame to a json file.
        """
        if not compression:
            compression_kwargs = None
        elif compression and compression_kwargs is None:
            compression_kwargs = {'method': 'gzip', 'compresslevel': 1, 'mtime': 1}

        self.results_df.to_json(filename, orient="columns", indent=1, compression=compression_kwargs)

    def train_test_split(self, train_size=0.8, rng=np.random.RandomState(42), discard_na_rows=False):
        """

        :param train_size: Fraction of the data to use for training
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        starting with "Success"
        :return:
        Train and test dataframes
        """
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        # split the results into train and test
        train_df = results_df.sample(frac=train_size, random_state=rng)
        test_df = results_df.drop(train_df.index)

        return train_df, test_df

    def train_val_test_split(self, train_size=0.8, val_size=0.1, rng=np.random.RandomState(42), discard_na_rows=False):
        """

        :param train_size: Fraction of the data to use for training
        :param val_size: Fraction of the data to use for validation
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        :return:
        Train, validation and test dataframes
        """
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        # split the results into train, validation and test
        train_df = results_df.sample(frac=train_size, random_state=rng)
        val_test_df = results_df.drop(train_df.index)
        val_df = val_test_df.sample(frac=val_size / (1 - train_size), random_state=rng)
        test_df = val_test_df.drop(val_df.index)

        return train_df, val_df, test_df

    def multiple_way_split(self, sizes=[0.4, 0.1, 0.4, 0.1], rng=np.random.RandomState(42), discard_na_rows=False):
        """
        :param sizes: List of fractions of the data to use for the 4 splits
        :param rng: RandomState object to use for reproducibility
        :param discard_na_rows: If True, discard all rows where there is at least one nan in the columns
        :return:
        Four dataframes with sizes corresponding to those in sizes
        """
        assert sum(sizes) == 1, "The sum of the sizes must be 1"
        results_df = self.discard_if_one_na_in_success_per_row() if discard_na_rows else self.results_df
        n_samples = len(results_df)
        # create size of splits
        splits = [int(size * n_samples) for size in sizes]
        if sum(splits) != n_samples:
            splits[-1] += n_samples - sum(splits)
        # create an array of 0,1,2,3 with sizes corresponding to the splits
        split_array = np.concatenate([np.full(size, i) for i, size in enumerate(splits)])
        # shuffle the array
        rng.shuffle(split_array)
        split_dataframes = []
        for i in range(4):
            split_dataframes.append(results_df.iloc[split_array == i])
        return split_dataframes

    def discard_if_one_na_in_success_per_row(self, inplace=False, print_n_discarded=False):
        # discard all rows where there is at least one nan in the "Success" columns for the considered llms
        length_before = len(self.results_df)
        res_df_after_discard = self.results_df[
            self.results_df[[f"Success_{llm}" for llm in self.llms]].notna().all(axis=1)]
        if print_n_discarded:
            print(f"Discarded {length_before - len(res_df_after_discard)} rows")
        if inplace:
            self.results_df = res_df_after_discard
            return self
        else:
            return res_df_after_discard

    def shuffle_data(self, inplace=True, rng=np.random.RandomState(42)):
        if inplace:
            self.results_df = self.results_df.sample(frac=1, random_state=rng)
            return self
        else:
            return self.results_df.sample(frac=1, random_state=rng)

    def get_average_success(self, llm):
        # check that column exists in the first place
        if f"Success_{llm}" not in self.results_df.columns:
            raise ValueError(f"Column Success_{llm} does not exist in the dataframe")

        # discard rows where that llm has not been tested when computing the mean
        return self.results_df[self.results_df[f"Success_{llm}"].notna()][f"Success_{llm}"].mean()

    def _extract_prompts_df(self, features_to_add=None, max_n_samples=None, skip_na_rows=False,
                            add_system_prompt=False):
        if features_to_add is None:
            features_to_add = []
        if max_n_samples is None:
            max_n_samples = len(self.results_df)

        if skip_na_rows:
            prompts_df = self.discard_if_one_na_in_success_per_row()[["prompt"] + features_to_add]
        else:
            prompts_df = self.results_df[["prompt"] + features_to_add]

        if len(prompts_df) == 0:
            raise ValueError("No rows to annotate as there are no prompts in the dataframe shared for all llms.")

        prompts_df = prompts_df.iloc[:max_n_samples]

        # if add_system_prompt, add the system prompt to the dataframe
        if add_system_prompt:
            prompts_df["prompt_sys_prompt"] = prompts_df["prompt"].apply(lambda x: f"{self.system_prompt} {x}")

        return prompts_df

    def _check_system_prompt_exists(self, add_system_prompt):
        if add_system_prompt:
            if not hasattr(self, "system_prompt"):
                raise ValueError("System prompt has not been defined. Please define it before running this method.")

    def _print_if_verbose(self, msg):
        if self.verbose:
            print(msg)

    def _print_df_size(self, df, llm, type: str = "raw"):
        self._print_if_verbose(
            f"{'Loaded raw' if type == 'raw' else 'Created merged'} data for {llm}; shape {df.shape}")


class EvalsResultsLoader(ResultsLoader):
    """Generic class that loads the results of a dataset evaluated with evals. """

    base_path_raw_default = "../results/"

    def __init__(self, task, llms=None, base_path_raw=None, processed_filename=None, verbose=False,
                 load_results_kwargs={}, load_processed_kwargs={}):
        self.task = task
        if base_path_raw is None:
            base_path_raw = self.base_path_raw_default
        # determine self.default_llms from the files in the f"{base_path}/{self.task}" directory
        if not hasattr(self, "default_llms") and llms is None:
            self.default_llms = self._get_default_llms(base_path_raw)
            # print(f"Default llms for {self.task}: {self.default_llms}")

        super().__init__(llms=llms, base_path_raw=base_path_raw, processed_filename=processed_filename, verbose=verbose,
                         load_results_kwargs=load_results_kwargs, load_processed_kwargs=load_processed_kwargs)

        # extract the system prompt by finding the first element where "sampling" is not None
        first_sampling_not_none = self.results_df["sampling"].notna().idxmax()
        sampling = self.results_df["sampling"].loc[first_sampling_not_none]

        # NOTICE the following thing works because all of the "evals" datasets contain a system prompt, otherwise it
        # may give errors.
        if isinstance(sampling["prompt"], list):
            self.system_prompt = sampling["prompt"][0]["content"]
            # print("prompt is a list")
        else:
            self.system_prompt = sampling["prompt"].split("\nUser: ")[
                0]

        # now can drop the "match" columns
        if "match" in self.results_df.columns:
            self.results_df = self.results_df.drop(columns=["match"])
        # do not drop sampling as this may be needed later on (for instance if you save the df and reload)
        # if "sampling" in self.results_df.columns:
        #     self.results_df = self.results_df.drop(columns=["sampling"])

    def load_results(self, base_path="../results/"):
        llms = self.llms

        results_dict = {}
        for llm in llms:
            results_dict[llm] = {}
            raw_data = pd.read_json(f"{base_path}/{self.task}/{llm}.jsonl", lines=True)

            results_df = self.reformat_results_df(raw_data)
            # check if there are duplicated entries:
            duplicated = results_df.duplicated(subset=["prompt"])
            if duplicated.any():
                self._print_if_verbose(f"Warning: {duplicated.sum()} duplicated entries for {llm}")
                # drop the duplicates?
                results_df = results_df.drop_duplicates(subset=["prompt"])

            results_dict[llm] = results_df
            # print("llm", llm)
            # print("accuracy rate ", results_df["Success"].sum() / len(results_df), len(results_df))
            # print("conform answers ", results_df["conform_answer"].sum() / len(results_df))
            self._print_df_size(results_dict[llm], llm, type="raw")
            # rename the "Success" column to "Success_{llm}"
            results_dict[llm] = results_dict[llm].rename(
                columns={"Success": f"Success_{llm}", "Answer": f"Answer_{llm}"})

        # now I want to create a single dataframe with all the results. In practice, start from the first and append
        # the "success" columns of the other llms, making sure that they go with the same prompt
        results_df = results_dict[llms[0]]
        self._print_df_size(results_df, llms[0], type="merged")
        for llm in llms[1:]:
            results_df = results_df.merge(results_dict[llm][["prompt", f"Success_{llm}", f"Answer_{llm}"]],
                                          on="prompt", how="outer")
            self._print_df_size(results_df, llm, type="merged")

        return results_df

    def reformat_results_df(self, results_df):
        # discard all things where run_id is nan
        results_df = results_df[~results_df["run_id"].isna()]
        # drop the "spec", "final_report" and "created_by" columns
        results_df = results_df.drop(columns=["spec", "final_report", "created_by"], errors="ignore")

        # There are two events for each "sample_id" (one for the sampling and one for the evaluation, called "match").
        # Use pivot to put them in a proper format
        # results dataframe in a different format: one row per sample_id, where I will store the "data" entry for the "sampling" and "metrics" events
        results_df = results_df.pivot(index="sample_id", columns="type", values="data")
        # define "Success"
        # results_df["Success"] = results_df.apply(lambda x: 1 if x["match"]["correct"] else 0, axis=1)
        # IMPORTANT I redefine here success as the naive match does not strip nor lower the strings
        results_df["Success"] = results_df.apply(self.check_answer_correct, axis=1)
        results_df["Answer"] = results_df.apply(self.extract_model_answer, axis=1)
        # now extract the original prompt; this depends on whether the prompt is a list or not
        # the way the prompt is stored is different for chat and completion models
        if isinstance(results_df.iloc[0]["sampling"]["prompt"], list):
            results_df["prompt"] = results_df.apply(lambda x: x["sampling"]["prompt"][1]["content"], axis=1)
            # print("prompt is a list")
        else:
            results_df["prompt"] = results_df.apply(lambda x: x["sampling"]["prompt"], axis=1)
            # notice this also discards the few-shot prompt
            results_df["prompt"] = results_df["prompt"].apply(
                lambda x: x.split("\nUser: ")[-1].split("\nAssistant:")[0])
            # print("prompt is a string")

        return results_df

    @staticmethod
    def extract_model_answer(row):
        return row["match"]["sampled"].strip().lower().replace("\u00A0", " ")

    @staticmethod
    def check_answer_correct(row):
        # strip spaces, new lines and "."; also remove non-breaking spaces
        if isinstance(row["match"]["expected"], str):
            expected = row["match"]["expected"].strip().lower().replace("\u00A0", " ")
        elif isinstance(row["match"]["expected"], int):
            expected = str(row["match"]["expected"])
        else:
            raise ValueError(f"Expected is neither a string nor an integer: {row['match']['expected']}")
        sampled = row["match"]["sampled"].strip().lower().replace("\u00A0", " ")

        # The overall regular expression is constructed to find the expected string in a case-insensitive manner
        # within the sampled string, considering word boundaries and whitespace. The re.search function returns a
        # match object if a match is found, and None otherwise.

        match = re.search(
            r"(^|\b)" + re.escape(expected) + r"(\b|$|\s)",
            sampled,
        )

        correctness_regex = 1 if match else 0

        # correctness_trivial = 1 if expected in sampled else 0
        #
        # # print the two if they are different
        # if correctness_regex != correctness_trivial:
        #     print("==== Correctness: regex=", correctness_regex, "trivial=", correctness_trivial)
        #     print(f'expected>{row["match"]["expected"]}---')
        #     print(f'sampled>{row["match"]["sampled"]}---')

        return correctness_regex

        # if expected in sampled:
        #     return 1
        # else:
        #     return 0

    def _get_default_llms(self, base_path_raw):
        # check in the directory of the task for all *jsonl files and extract the llms from the filenames
        # list of all files
        all_files = os.listdir(f"{base_path_raw}/{self.task}")
        # list of all jsonl files
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
        # list of all llms
        llms = [f.replace(".jsonl", "") for f in jsonl_files]
        # now sort the list of llms using sort_models_order
        llms = [llm for llm in sort_models_order if llm in llms]
        return llms


def _finalize_train_validation_test_dfs(train_df, test_df, validation_size, subsampled_n_train,
                                        subsampled_n_test, random_state):
    # subsample
    if subsampled_n_train is not None:
        train_df = train_df.sample(n=subsampled_n_train, random_state=random_state)
    if subsampled_n_test is not None:
        test_df = test_df.sample(n=subsampled_n_test, random_state=random_state)

    # split the training set into train and validation
    validation_df = train_df.sample(frac=validation_size, random_state=random_state)
    train_df = train_df.drop(validation_df.index)

    # reset all indices
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, validation_df, test_df


def load_kinds_of_reasoning(llms, base_path="2_results/", validation_size=0.2,
                   subsampled_n_train=None, subsampled_n_test=None, random_state=42):
    """
    This loads the KindsOfReasoning evaluation results and creates the final files to be released.

    The final dataset has different train-validation-test Splits:
    Random. Random shuffle split (in-distribution)
    OOD_1. Arithmetic left out
    OOD_2. Causal reasoning left out
    OOD_3. all the other kinds of reasoning left out
    OOD_4. world knowledge and common sense left out

    Notice that splits 1, 2 and 3 are a covering of the full dataset; the final split is instead complementary.

    :param llms: list of llms to consider
    :param base_path:
    :param validation_size: dedicate that fraction of the train split to validation. This is done after subsampling
    the training set (see below)
    :param subsampled_n_train: If not None, the train split will be subsampled to this number of rows
    :param subsampled_n_test: If not None, the test split will be subsampled to this number of rows
    :param random_state: random state for the subsampling

    :return:
    A single dataframe containing all prompts and system prompts, the success and the answers for all llms, and
    the various splits ("train", "validation", "test" for the different splits).
    """

    evals_dict = {
        'formal_fallacies_syllogisms_negation': ["logical_reasoning"],
        'logical_args': ["logical_reasoning", "common_sense"],
        'babi_task_16': ["inductive_reasoning"],
        'logiqav2-logical-reasoning-plus': ["deductive_reasoning"],
        'wanli': ["deductive_reasoning"],
        'alpha_nli': ["abductive_reasoning"],
        'reclor-logical-reasoning-plus': ["abductive_reasoning", "deductive_reasoning", "inductive_reasoning"],
        'crass_ai': ["counterfactual_reasoning"],
        'cause_and_effect_one_sentence': ["causal_reasoning"],
        'cause_and_effect_two_sentences': ["causal_reasoning"],
        'fantasy_reasoning': ["causal_reasoning"],
        'goal_step_wikihow_goal_inference': ["causal_reasoning"],
        'goal_step_wikihow_step_inference': ["causal_reasoning"],
        'goal_step_wikihow_step_ordering': ["causal_reasoning"],
        "copa": ["causal_reasoning", "world_knowledge"],
        "cosmos_qa": ["causal_reasoning", "world_knowledge"],
        "ropes": ["causal_reasoning", "world_knowledge"],
        "anli": ["causal_reasoning", "world_knowledge"],
        "emoji_movie": ["analogical_reasoning", "world_knowledge"],
        'abstract_narrative_understanding_9_distractors': ["analogical_reasoning"],
        'abstract_narrative_understanding_99_distractors': ["analogical_reasoning"],
        'odd_one_out': ["analogical_reasoning"],
        'metaphor_boolean': ["analogical_reasoning"],
        'geometric_shapes': ["spatial_reasoning"],
        'space_nli': ["spatial_reasoning"],
    }

    # add the aritmetic results:
    for n_digits in range(1, 6):
        for operation in ["addition", "subtraction", "multiplication", "division"]:
            eval_name = f"arithmetic_{n_digits}_digit_{operation}"
            evals_dict[eval_name] = ["arithmetic"]

    splits_dict = {
        1: ["arithmetic"],
        2: ["causal_reasoning"],
        3: ["logical_reasoning", "deductive_reasoning", "inductive_reasoning", "spatial_reasoning",
                  "abductive_reasoning", "counterfactual_reasoning", "analogical_reasoning"],
        4: ["world_knowledge", "common_sense"],
    }

    loaded_evals_dict = {}
    # load all datasets (to avoid loading multiple times)
    for i, eval_name in tqdm(enumerate(evals_dict.keys())):
        instance = EvalsResultsLoader(task=eval_name, llms=llms, base_path_raw=base_path)
        instance = instance.discard_if_one_na_in_success_per_row(
            inplace=True)  # this discards all rows where the considered llm has not been tested
        instance.results_df["system_prompt"] = instance.system_prompt
        # discard "sampling" from instance.results_df
        instance.results_df = instance.results_df.drop(columns=["sampling"])
        instance.results_df["dataset"] = eval_name
        loaded_evals_dict[eval_name] = instance.results_df

    final_df = None

    # do loop on all splits and generate a df for each of them; then eventually merge all of them
    for split in [False, 1, 2, 3, 4]:

        if split:
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
        else:
            pooled_results_df = pd.DataFrame()

        for i, eval_name in tqdm(enumerate(evals_dict.keys())):

            if split:
                if any([kind in splits_dict[split] for kind in evals_dict[eval_name]]):
                    test_df = pd.concat([test_df, loaded_evals_dict[eval_name]], ignore_index=True)
                else:
                    train_df = pd.concat([train_df, loaded_evals_dict[eval_name]], ignore_index=True)
            else:
                # add this to the pooled results
                pooled_results_df = pd.concat([pooled_results_df, loaded_evals_dict[eval_name]], ignore_index=True)

        if not split:
            # now do a train-test split
            train_df = pooled_results_df.sample(frac=0.7, random_state=random_state)
            test_df = pooled_results_df.drop(train_df.index)

        train_df, validation_df, test_df = _finalize_train_validation_test_dfs(
            train_df, test_df, validation_size, subsampled_n_train, subsampled_n_test, random_state)

        # now recombine into a single dataframe
        split_name = f"OOD_{split}_split" if split else "Random_split"
        train_df[split_name] = "train"
        validation_df[split_name] = "validation"
        test_df[split_name] = "test"

        single_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

        # merge by using the prompt
        if final_df is not None:
            # need only to add the split_name column
            single_df = single_df[["prompt", split_name]]

            final_df = pd.merge(final_df, single_df, on="prompt", how="outer")
        else:
            final_df = single_df

    # sort the columns: first prompt and system prompt, then the various splits, then all the success and answers:
    columns_before = final_df.columns
    final_df = final_df[["prompt", "system_prompt", "dataset", "Random_split"] + [f"OOD_{i+1}_split" for i in range(4)] +
                        [f"Success_{llm}" for llm in llms] + [f"Answer_{llm}" for llm in llms]]
    columns_after = final_df.columns

    return final_df


if __name__ == "__main__":
    llms_reasoning = [
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
    ]

    final_df = load_kinds_of_reasoning(llms=llms_reasoning, validation_size=0.2,
                              subsampled_n_train=None, subsampled_n_test=None, random_state=42)

    # save the file into a csv:
    final_df.to_csv("../KindsOfReasoning_with_llm_results.csv", index=False)
    # save the file into a json
    final_df.to_json("../KindsOfReasoning_with_llm_results.json", orient="records", lines=True)

    final_df_without_results = final_df[["prompt", "system_prompt", "dataset", "Random_split"] + [f"OOD_{i+1}_split" for i in range(4)]]

    # save the file into a csv:
    final_df_without_results.to_csv("../KindsOfReasoning.csv", index=False)
    # save the file into a json
    final_df_without_results.to_json("../KindsOfReasoning.json", orient="records", lines=True)