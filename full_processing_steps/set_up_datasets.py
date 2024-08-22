import argparse
import json
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Callable, TypeVar

import datasets as datasets_hf_lib
import numpy
import numpy as np
import pandas as pd
import yaml

from src.downloader_utils import ensure_file_downloaded


class DatasetPreparerAbstract(ABC):

    def __init__(self, num_few_shot=4):
        self.num_few_shot = num_few_shot

    def download_raw_data(
            self,
            target_path: str,
    ):
        # if no unpack, then add a filename with the right extension. Use the eval_id as the filename
        # get the extension from the source_url
        if isinstance(self.source_url, list):
            # check if self has attribute "subtasks" which is a list as well with the same length as self.source_url
            if not hasattr(self, "subtasks"):
                raise ValueError("If source_url is a list, then subtasks must be defined as well")
            # check if subtasks is a list:
            if not isinstance(self.subtasks, list):
                raise ValueError("If source_url is a list, then subtasks must be a list as well")
            if len(self.source_url) != len(self.subtasks):
                raise ValueError("If source_url is a list, then subtasks must have the same length")
            source_url = self.source_url
            subtasks = self.subtasks
        else:
            # wrap the source_url in a list
            # notice that this also works if there are multiple subtasks using the same data
            source_url = [self.source_url]
            subtasks = [None]
        for subtask, source_url in zip(subtasks, source_url):

            if not self.unpack:
                # find extension:
                # unless there is "format" towards the end of the string, the extension is the last part of the string
                if "format" not in source_url.split("/")[-1]:
                    extension = source_url.split(".")[-1]
                else:
                    # extension comes after format=
                    extension = source_url.split("format=")[-1]

                target_path_2 = target_path + "/" + self.eval_id
                if subtask is not None:
                    target_path_2 = target_path_2 + "_" + subtask

                target_path_2 += "." + extension
            else:
                target_path_2 = target_path

            ensure_file_downloaded(
                source_url=source_url,
                target_path=target_path_2,
                unpack=self.unpack,
                unpack_type=self.unpack_type,
            )

    @abstractmethod
    def transform_raw_data(self, path_to_raw_data, registry_data_path, rng_seed):
        """This function is specific to each dataset"""
        pass

    @staticmethod
    def create_chat_prompt_multiple_choice(sys_msg, question, choices, answers):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def create_chat_prompt(sys_msg, question):
        user_prompt = f"{question}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def create_chat_example_multiple_choice(question, choices, answers, correct_answer):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        return [
            {"role": "system", "content": user_prompt, "name": "example_user"},
            {"role": "system", "content": correct_answer, "name": "example_assistant"},
        ]

    @staticmethod
    def create_chat_example(question, correct_answer):
        user_prompt = f"{question}"
        return [
            {"role": "system", "content": user_prompt, "name": "example_user"},
            {"role": "system", "content": correct_answer, "name": "example_assistant"},
        ]

    def register_dataset(self, registry_path):

        type = self.eval_template

        class_dict = {
            "match": "evals.elsuite.basic.match:Match",
            "includes": "evals.elsuite.basic.includes:Includes",
            "fuzzy_match": "evals.elsuite.basic.fuzzy_match:FuzzyMatch",
            "fact": "evals.elsuite.modelgraded.classify:ModelBasedClassify"
        }

        if type not in class_dict.keys():
            raise ValueError(f"Type {type} not recognized. Must be one of match, includes, fuzzy_match or fact")

        if not hasattr(self, "subtasks"):
            subtasks = [None]
        elif isinstance(self.subtasks, list):
            subtasks = self.subtasks
        else:
            subtasks = [self.subtasks]

        registry_yaml = {}

        for subtask in subtasks:
            eval_id = self.eval_id

            if subtask is not None:
                eval_id = eval_id + "_" + subtask

            registry_yaml[eval_id] = {
                "id": f"{eval_id}.test.v1",
                "metrics": ["accuracy"]
            }
            registry_yaml[f"{eval_id}.test.v1"] = {
                "class": class_dict[type],
                "args": {
                    "samples_jsonl": self._get_samples_path(subtask=subtask),
                }
            }

            if type == "fact":
                registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_type"] = "cot_classify"
                registry_yaml[f"{eval_id}.test.v1"]["args"]["modelgraded_spec"] = type
                if hasattr(self, "eval_completion_fn"):
                    registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_completion_fn"] = self.eval_completion_fn

            # now the few shot part:
            eval_id = eval_id + "_few_shot"

            registry_yaml[eval_id] = {
                "id": f"{eval_id}.test.v1",
                "metrics": ["accuracy"]
            }
            registry_yaml[f"{eval_id}.test.v1"] = {
                "class": class_dict[type],
                "args": {
                    "samples_jsonl": self._get_samples_path(subtask=subtask),
                    "few_shot_jsonl": self._get_samples_path(few_shot=True, subtask=subtask),
                    "num_few_shot": self.num_few_shot,
                }
            }

            if type == "fact":
                registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_type"] = "cot_classify"
                registry_yaml[f"{eval_id}.test.v1"]["args"]["modelgraded_spec"] = type
                if hasattr(self, "eval_completion_fn"):
                    registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_completion_fn"] = self.eval_completion_fn

        self._save_yaml_registry(registry_path, registry_yaml)

        # this is actually not needed
        # if type == "fact":
        #     # then need to store the fact.yaml file in the modelgraded folder
        #     url = "https://github.com/openai/evals/raw/4b7a66bd45f06156656e021e170e7574f6cde3f5/evals/registry/modelgraded/fact.yaml"
        #     file_path = self._get_modelgraded_yaml_path(registry_path, type)
        #     # download the file in the right place
        #     ensure_file_downloaded(
        #         source_url=url,
        #         target_path=file_path,
        #         unpack=False,
        #         unpack_type=None,
        #     )

    def _get_samples_path(self, registry_path=None, subtask=None, few_shot=False):
        path_elements = [self.eval_id]

        if subtask is not None:
            path_elements += [subtask]

        if registry_path is not None:
            path_elements = [registry_path, "data"] + path_elements
            # create the folders if they do not exist
            os.makedirs(os.path.join(*path_elements), exist_ok=True)

        path_elements += ["samples.jsonl" if not few_shot else "few_shot.jsonl"]

        return os.path.join(*path_elements)

    def _get_yaml_path(self, registry_path):
        # create the folders if they do not exist
        os.makedirs(os.path.join(registry_path, "evals"), exist_ok=True)
        return os.path.join(registry_path, "evals", f"{self.eval_id}.yaml")

    @staticmethod
    def _get_modelgraded_yaml_path(registry_path, type):
        # create the folders if they do not exist
        os.makedirs(os.path.join(registry_path, "modelgraded"), exist_ok=True)
        return os.path.join(registry_path, "modelgraded", f"{type}.yaml")

    def _save_yaml_registry(self, registry_path, registry_yaml):
        with open(self._get_yaml_path(registry_path), "w") as f:
            yaml.dump(registry_yaml, f)


class DatasetPreparerFromURL(DatasetPreparerAbstract, ABC):
    source_url: str
    unpack: bool
    unpack_type: str
    eval_id: str
    eval_template: str

    @classmethod
    def __init_subclass__(cls):
        """This is run before the init of the subclass and raises an error if the class variables are not defined in the
        subclass"""
        required_class_variables = [
            'source_url',
            'unpack',
            'unpack_type',
            'eval_id',
            'eval_template',
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
        # check if the eval_template is valid
        if cls.eval_template not in ["match", "includes", "fuzzy_match", "fact"]:
            raise ValueError(f"Eval template {cls.eval_template} not recognized. Must be one of match, includes, "
                             f"fuzzy_match or fact")


class DatasetPreparerHuggingFace(DatasetPreparerAbstract, ABC):
    """The only difference here will be the download_raw_data function, which will use the HuggingFace datasets
    library. For this reason, the arguments have changed too"""
    huggingface_name: str
    huggingface_split: str
    eval_id: str
    eval_template: str

    def __init_subclass__(cls):
        """This is run before the init of the subclass and raises an error if the class variables are not defined in the
        subclass"""
        required_class_variables = [
            'huggingface_name',
            'huggingface_split',
            'eval_id',
            'eval_template',
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
        # check if the eval_template is valid
        if cls.eval_template not in ["match", "includes", "fuzzy_match", "fact"]:
            raise ValueError(f"Eval template {cls.eval_template} not recognized. Must be one of match, includes, "
                             f"fuzzy_match or fact")

    def download_raw_data(
            self,
            target_path: str,
    ):
        self.hf_dataset = datasets_hf_lib.load_dataset(self.huggingface_name, self.huggingface_split)


class DatasetPreparerBIGBench(DatasetPreparerAbstract, ABC):
    """This only work for json tasks which are multiple choice."""

    eval_id: str
    eval_template: str
    subtasks: str
    sys_msg: str
    ideal_index: bool
    unpack = False
    unpack_type = None
    allowed_tasks = [
        "formal_fallacies_syllogisms_negation",
        "logical_args",
        "crass_ai",
        "mnist_ascii",
        "geometric_shapes",
        "emoji_movie",
        "odd_one_out",
        "metaphor_boolean",
        "causal_judgment",
        "fantasy_reasoning",
        "moral_permissibility",
        "crash_blossom",
        "sentence_ambiguity",
        "dyck_languages",
        # the following are those with subtasks
        "intersect_geometry",
        "symbol_interpretation",
        "abstract_narrative_understanding",
        "conceptual_combinations",
        "cause_and_effect",
        "goal_step_wikihow",
        "arithmetic",
    ]

    @classmethod
    def __init_subclass__(cls):
        """This is run before the init of the subclass and raises an error if the class variables are not defined in the
        subclass"""
        required_class_variables = [
            'eval_id',
            'eval_template',
            'subtasks',
            'sys_msg',
            'ideal_index'
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
        # check if the eval_template is valid
        if cls.eval_template not in ["match", "includes", "fuzzy_match", "fact"]:
            raise ValueError(f"Eval template {cls.eval_template} not recognized. Must be one of match, includes, "
                             f"fuzzy_match or fact")

        if cls.eval_id not in cls.allowed_tasks:
            raise ValueError(f"Eval id {cls.eval_id} not recognized. Must be one of {cls.allowed_tasks}")

    def __init__(self, num_few_shot=4):
        self.source_url = self._get_json_raw_url()
        super().__init__(num_few_shot=num_few_shot)

    def _get_json_raw_url(self, commit_n="6436ed17f979b138463224421f8e8977b89076ed"):
        urls = []
        if self.subtasks is not None:
            for subtask in self.subtasks:
                urls.append(
                    f"https://raw.githubusercontent.com/google/BIG-bench/{commit_n}/bigbench/benchmark_tasks/{self.eval_id}/{subtask}/task.json")
        else:
            urls = f"https://raw.githubusercontent.com/google/BIG-bench/{commit_n}/bigbench/benchmark_tasks/{self.eval_id}/task.json"
        return urls

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):

        rng = numpy.random.default_rng(rng_seed)
        rng2 = numpy.random.default_rng(rng_seed + 1)

        if isinstance(self.subtasks, list):
            subtasks = self.subtasks
        else:
            subtasks = [self.subtasks]

        for subtask in subtasks:
            if subtask is None:
                data_path = os.path.join(path_to_raw_data, f"{self.eval_id}.json")
            else:
                data_path = os.path.join(path_to_raw_data, f"{self.eval_id}_{subtask}.json")

            # load the dict from the json file
            with open(data_path, "r") as f:
                data = json.load(f)

            append_choices_to_input = True
            if "append_choices_to_input" in data.keys():
                append_choices_to_input = data["append_choices_to_input"]
            if hasattr(self, "append_choices_to_input"):
                # overwrite
                append_choices_to_input = self.append_choices_to_input

            if "task_prefix" in data.keys():
                if self.sys_msg == "":
                    sys_msg = data["task_prefix"]
                else:
                    sys_msg = data["task_prefix"].strip() + " " + self.sys_msg.strip()
            else:
                sys_msg = self.sys_msg

            # the samples are here:
            df = pd.DataFrame(data["examples"])

            # shuffle
            df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

            # some processing which is identical for all tasks

            # convert the target_scores (now a dict) to a list of tuples
            df["target_scores"] = df["target_scores"].apply(lambda x: list(x.items()))

            # shuffle
            df["target_scores"] = df["target_scores"].apply(lambda x: rng.permutation(x).tolist())

            # extract the choices from the target_scores
            df["choices"] = df["target_scores"].apply(lambda x: [y[0] for y in x])

            # extract the correct answer
            df["correct_answer"] = df["target_scores"].apply(
                lambda x: list([y[1] for y in x]).index(max([y[1] for y in x])))

            if self.ideal_index:
                df["ideal"] = df.apply(lambda x: str(x["correct_answer"] + 1), axis=1)
            else:
                df["ideal"] = df.apply(lambda x: str(x["choices"][x["correct_answer"]]), axis=1)

            # then do the task specific transformation
            df = self.task_specific_transformation(df)

            # now split into few shot and test
            if self.num_few_shot > 0:
                few_shot_df = df[:self.num_few_shot]
                # take those away from the df
                df = df[self.num_few_shot:]

                if append_choices_to_input:
                    # check if all "ideal" are the same in few_shot_df
                    if self.ideal_index and few_shot_df["ideal"].nunique() == 1:
                        # if so, we can permute the choices
                        for i, row in few_shot_df.iterrows():
                            # generate permutation indices of choices
                            perm = rng2.permutation(len(row["choices"]))
                            # apply the permutation to the choices
                            few_shot_df.at[i, "choices"] = [row["choices"][j] for j in perm]
                            # update the ideal index:
                            few_shot_df.at[i, "ideal"] = str(np.argwhere(perm == int(row["ideal"]) - 1)[0][0] + 1)

                    few_shot_df["sample"] = few_shot_df.apply(
                        lambda x: self.create_chat_example_multiple_choice(
                            x["input"],
                            (np.arange(len(x["choices"])) + 1).tolist(),
                            x['choices'], x["ideal"]), axis=1)
                else:
                    few_shot_df["sample"] = few_shot_df.apply(
                        lambda x: self.create_chat_example(
                            x["input"], x["ideal"]), axis=1)

                cols_to_save = ["sample"]

                # add the "comment" column if it exists
                if "comment" in few_shot_df.columns:
                    cols_to_save.append("comment")

                few_shot_df[cols_to_save].to_json(self._get_samples_path(registry_path, subtask=subtask, few_shot=True),
                                                  lines=True, orient="records")

            if append_choices_to_input:
                df["input"] = df.apply(
                    lambda x: self.create_chat_prompt_multiple_choice(sys_msg, f"{x['input']}",
                                                                      (np.arange(len(x["choices"])) + 1).tolist(),
                                                                      x['choices']), axis=1)
            else:
                df["input"] = df.apply(lambda x: self.create_chat_prompt(sys_msg, f"{x['input']}"), axis=1)

            cols_to_save = ["input", "ideal"]

            # add the "comment" column if it exists
            if "comment" in df.columns:
                cols_to_save.append("comment")

            df[cols_to_save].to_json(self._get_samples_path(registry_path, subtask=subtask), lines=True,
                                     orient="records")

    @staticmethod
    def task_specific_transformation(df):
        return df


# --- This part of code is adapted from the HELM repository: https://github.com/stanford-crfm/helm,
# which is released under Apache License 2.0 ---
DATASETS_DICT: Dict[str, DatasetPreparerAbstract] = {}
"""Dict of dataset names (or ids) to DatasetPreparerAbstract classes."""

F = TypeVar("F", bound=Callable[..., DatasetPreparerAbstract])


def dataset_class(name: str) -> Callable[[F], F]:
    """Register the run spec function under the given name."""

    def wrap(dataset: F) -> F:
        if name in DATASETS_DICT:
            raise ValueError(f"A dataset with name {name} already exists")
        DATASETS_DICT[name] = dataset
        return dataset

    return wrap


# --- End of adapted code ---

# --- The following datasets are part of the KindsOfReasoning collection ---


@dataset_class("space_nli")
class SpaceNLI(DatasetPreparerFromURL):
    source_url = "https://raw.githubusercontent.com/kovvalsky/SpaceNLI/8c10e94d238737be97142f5a7ffdffa49a6a6ab9/dataset/160x200.json"
    unpack = False
    unpack_type = None
    eval_id = "space_nli"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42, n_samples_per_problem=10):
        sys_msg = ("I will provide you with a premise and a hypothesis. You have to answer whether the relation "
                   "between premise and hypothesis is entailment, neutral or contradiction. Answer by only using"
                   " the words 'entailment', 'neutral' or 'contradiction'.")
        choices = ["E", "N", "C"]
        answers = ["entails", "neutral to", "contradicts"]
        answers_2 = ["entailment", "neutral", "contradiction"]
        answers_dict = {answer: choice for choice, answer in zip(choices, answers_2)}

        data_path = os.path.join(path_to_raw_data, f"{self.eval_id}.json")  # this will most likely not work

        df = pd.read_json(data_path)
        df = df.join(pd.json_normalize(df.data))
        # now only keep the useful cols
        df = df[["id", "label", "prem_num", "premises", "hypothesis"]]
        # simplify the id
        df["problem_id"] = df["id"].apply(lambda x: x.split("-")[0])
        df["sample_id"] = df["id"].apply(lambda x: x.split("-")[1])

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        rng = numpy.random.default_rng(rng_seed)

        if self.num_few_shot > 0:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(
                lambda x: self.create_chat_example(
                    f"Premise: {x['premises']}\nHypothesis: {x['hypothesis']}",
                    x["label"]),
                axis=1)
            few_shot_df[["sample", "problem_id"]].to_json(self._get_samples_path(registry_path, few_shot=True),
                                                          lines=True, orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt(
                sys_msg,
                f"Premise: {x['premises']}\nHypothesis: {x['hypothesis']}",
            ), axis=1)
        df["ideal"] = df.label

        # now select only a subset of samples per problem and shuffle them
        # make the sampling reproducible
        df = df.groupby("problem_id").apply(lambda x: x.sample(n_samples_per_problem, random_state=rng)).reset_index(
            drop=True)
        # save
        df[["input", "ideal", "problem_id"]].to_json(self._get_samples_path(registry_path), lines=True,
                                                     orient="records")


@dataset_class("anli")
class ANLI(DatasetPreparerFromURL):
    source_url = "https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip"
    unpack = True
    unpack_type = None
    eval_id = "anli"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = ("I will provide you with a context and a hypothesis. You have to answer "
                   "whether the relation between context and hypothesis is entailment, neutral or contradiction. "
                   "Answer by only using the words 'entailment', 'neutral' or 'contradiction'.")

        choices = ["e", "n", "c"]
        answers = ["entailment", "neutral", "contradiction"]
        answers_dict = {choice: answer for choice, answer in zip(choices, answers)}

        anli_test_r1 = pd.read_json(path_to_raw_data + '/R1/test.jsonl', lines=True)
        anli_test_r2 = pd.read_json(path_to_raw_data + '/R2/test.jsonl', lines=True)
        anli_test_r3 = pd.read_json(path_to_raw_data + '/R3/test.jsonl', lines=True)

        # concatenate the three test sets
        df = pd.concat([anli_test_r1, anli_test_r2, anli_test_r3], ignore_index=True)

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                f"Context: {x['context']}\nHypothesis: {x['hypothesis']}", answers_dict[x["label"]]), axis=1)
            few_shot_df[["sample", "model_label", "emturk", "genre", "reason", "tag"]].to_json(
                self._get_samples_path(registry_path, few_shot=True), lines=True,
                orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt(sys_msg,
                                              f"Context: {x['context'].strip()}\nHypothesis: {x['hypothesis'].strip()}"),
            axis=1)
        df["ideal"] = df["label"].apply(lambda x: answers_dict[x])

        # save
        df[["input", "ideal", "model_label", "emturk", "genre", "reason", "tag"]].to_json(
            self._get_samples_path(registry_path), lines=True, orient="records"
        )


@dataset_class("copa")
class COPA(DatasetPreparerFromURL):
    source_url = "https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz"  # todo looks like they've removed this link!
    unpack = True
    unpack_type = "untar"
    eval_id = "copa"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = "I will provide you with a fact and a question, with two possible answers (1 and 2).  You " \
                  "have to answer whether 1 or 2 is the correct answer. Answer by indicating '1' or '2' only."
        choices = ["1", "2"]

        data_path = os.path.join(path_to_raw_data, "datasets", f"{self.eval_id}-test.xml")

        # Parse the XML file
        tree = ET.parse(data_path)
        root = tree.getroot()

        # Create empty lists to store the data
        data = []

        # Extract data from XML and populate lists
        for item in root.findall('item'):
            # Access attributes of the 'item' element
            item_id = item.attrib['id']
            asks_for = item.attrib['asks-for']
            most_plausible_alternative = item.attrib['most-plausible-alternative']

            # Access text content of child elements
            p_text = item.find('p').text
            a1_text = item.find('a1').text
            a2_text = item.find('a2').text

            data.append([item_id, p_text, a1_text, a2_text, asks_for, most_plausible_alternative])

        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=['id', 'fact', 'a1', 'a2', 'asks_for', 'most_plausible_alternative'])

        df["text"] = df.apply(
            lambda x: "Fact: " + x[
                "fact"].strip() + f"\nQuestion: what is the most plausible {x['asks_for'].strip()} of this?",
            axis=1)

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            # make sure that there are the same number of 1's and 2's in most_plausible_alternative
            for i, row in few_shot_df.iterrows():
                if row["most_plausible_alternative"] == "1" and i % 2 == 0:
                    few_shot_df.at[i, "most_plausible_alternative"] = "2"
                    # swap a1 and a2
                    a1 = row["a1"]
                    few_shot_df.at[i, "a1"] = row["a2"]
                    few_shot_df.at[i, "a2"] = a1
                elif row["most_plausible_alternative"] == "2" and i % 2 == 1:
                    few_shot_df.at[i, "most_plausible_alternative"] = "1"
                    # swap a1 and a2
                    a1 = row["a1"]
                    few_shot_df.at[i, "a1"] = row["a2"]
                    few_shot_df.at[i, "a2"] = a1

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example_multiple_choice(
                x["text"], choices, [x["a1"], x["a2"]], x["most_plausible_alternative"]), axis=1)
            few_shot_df[["sample", "asks_for", "id"]].to_json(self._get_samples_path(registry_path, few_shot=True),
                                                              lines=True, orient="records")

        # Create test prompts and ideal completions
        df["input"] = df.apply(lambda x: self.create_chat_prompt_multiple_choice(
            sys_msg, x["text"], choices, [x["a1"], x["a2"]]), axis=1)
        df["ideal"] = df["most_plausible_alternative"]

        # save input, ideal and source
        df[["input", "ideal", "asks_for", "id"]].to_json(self._get_samples_path(registry_path), lines=True,
                                                         orient="records")


@dataset_class("alpha_nli")
class AlphaNLI(DatasetPreparerHuggingFace):
    huggingface_name = "logicreasoning/logi_glue"
    huggingface_split = "alpha_nli"
    eval_id = "alpha_nli"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = ("I will provide you with two observations and then ask a question with two options (A and B). You "
                   "have to answer whether A or B is the correct answer. Answer by indicating 'A' or 'B'.")

        df = self.hf_dataset['test'].to_pandas()

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example_multiple_choice(
                f"Context: {x['context']}\nQuestion: {x['question']}",
                ['A', 'B'], x['choices'], 'A' if x["answer_choice"] == 1 else 'B')
                                                      , axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt_multiple_choice(sys_msg,
                                                              f"Context: {x['context']}\nQuestion: {x['question']}",
                                                              ['A', 'B'], x['choices']), axis=1)
        df["ideal"] = df["answer_choice"].apply(lambda x: 'A' if x == 1 else 'B')

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


@dataset_class("wanli")
class WANLI(DatasetPreparerHuggingFace):
    huggingface_name = "logicreasoning/logi_glue"
    huggingface_split = "wanli"
    eval_id = "wanli"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = ("I will provide you with a premise and a hypothesis. You have to answer whether the relation between"
                   " premise and hypothesis is entailment, neutral or contradiction. Answer by only using the words"
                   " 'entailment', 'neutral' or 'contradiction'.")

        df = self.hf_dataset['test'].to_pandas()

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                x['context'], x['answer_text']), axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        df["input"] = df.apply(lambda x: self.create_chat_prompt(sys_msg, x["context"]), axis=1)
        df["ideal"] = df["answer_text"]

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


@dataset_class("babi_task_16")
class bABItask16(DatasetPreparerHuggingFace):
    huggingface_name = "logicreasoning/logi_glue"
    huggingface_split = "babi_task_16"
    eval_id = "babi_task_16"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = ("I will provide you with a context including a series of statements from which you have to infer"
                   " the answer to the question. Answer with a single word.")

        df = self.hf_dataset['test'].to_pandas()

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                x['input'], x['answer_text']), axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        df["input"] = df.apply(lambda x: self.create_chat_prompt(sys_msg, x["input"]), axis=1)
        df["ideal"] = df["answer_text"]

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


@dataset_class("ropes")
class ROPES(DatasetPreparerHuggingFace):
    huggingface_name = "ropes"
    huggingface_split = None
    eval_id = "ropes"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = (
            "I will provide you with a context and then ask a question. Answer by only reporting the correct answer.")

        df = self.hf_dataset['validation'].to_pandas()

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                "Context: " + x['situation'] + "\nQuestion: " + x['question'], x["answers"]['text'][0]),
                                                      axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt(sys_msg, "Context: " + x['situation'] + "\nQuestion: " + x['question']),
            axis=1)
        df["ideal"] = df["answers"].apply(lambda x: x['text'][0])

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


@dataset_class("cosmos_qa")
class CosmosQA(DatasetPreparerHuggingFace):
    huggingface_name = "cosmos_qa"
    huggingface_split = None
    eval_id = "cosmos_qa"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = (
            "I will provide you with a context and then ask a question with four options (A, B, C and D). You "
            "have to answer whether A, B, C or D is the correct answer. Answer by indicating 'A', 'B', 'C' or 'D'.")

        df = self.hf_dataset['validation'].to_pandas()

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example_multiple_choice(
                "Context: " + x['context'] + "\nQuestion: " + x['question'], ['A', 'B', 'C', 'D'],
                [x['answer0'], x['answer1'], x['answer2'], x['answer3']], ['A', 'B', 'C', 'D'][x["label"]]), axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt_multiple_choice(sys_msg,
                                                              "Context: " + x['context'] + "\nQuestion: " + x[
                                                                  'question'], ['A', 'B', 'C', 'D'],
                                                              [x['answer0'], x['answer1'], x['answer2'], x['answer3']]),
            axis=1)
        df["ideal"] = df.apply(lambda x: ['A', 'B', 'C', 'D'][x["label"]], axis=1)

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


@dataset_class("formal_fallacies_syllogisms_negation")
class FormalFallaciesSyllogismsNegation(DatasetPreparerBIGBench):
    eval_id = "formal_fallacies_syllogisms_negation"
    eval_template = "match"
    subtasks = None
    sys_msg = "Answer only with 'valid' or 'invalid'."
    ideal_index = False


@dataset_class("logical_args")
class LogicalArgs(DatasetPreparerBIGBench):
    eval_id = "logical_args"
    eval_template = "match"
    subtasks = None
    sys_msg = "I will provide you with multiple choice options. Answer with the correct index (1, 2, 3, 4 or 5)."
    ideal_index = True


@dataset_class("crass_ai")
class CrassAI(DatasetPreparerBIGBench):
    eval_id = "crass_ai"
    eval_template = "match"
    subtasks = None
    sys_msg = "I will provide you a question and multiple choice options. Answer with the correct index (1, 2, 3, 4 or 5)."
    ideal_index = True


@dataset_class("geometric_shapes")
class GeometricShapes(DatasetPreparerBIGBench):
    eval_id = "geometric_shapes"
    eval_template = "match"
    subtasks = None
    sys_msg = "I will provide you with an SVG path and options for what it draws. Answer with the correct option."
    ideal_index = False


@dataset_class("emoji_movie")
class EmojiMovie(DatasetPreparerBIGBench):
    eval_id = "emoji_movie"
    eval_template = "match"
    subtasks = None
    sys_msg = "I will provide you a question and multiple choice options. Answer with the correct index (1, 2, 3, 4 or 5)."
    ideal_index = True


@dataset_class("odd_one_out")
class OddOneOut(DatasetPreparerBIGBench):
    eval_id = "odd_one_out"
    eval_template = "match"
    subtasks = None
    sys_msg = "Answer with a single word."
    ideal_index = False


@dataset_class("metaphor_boolean")
class MetaphorBoolean(DatasetPreparerBIGBench):
    eval_id = "metaphor_boolean"
    eval_template = "match"
    subtasks = None
    sys_msg = "Answer with 'True' or 'False'."
    ideal_index = False


@dataset_class("fantasy_reasoning")
class FantasyReasoning(DatasetPreparerBIGBench):
    eval_id = "fantasy_reasoning"
    eval_template = "match"
    subtasks = None
    sys_msg = "I will provide you with a question about a fantasy world. Answer with 'Yes' or 'No'."
    ideal_index = False


@dataset_class("abstract_narrative_understanding")
class AbstractNarrativeUnderstanding(DatasetPreparerBIGBench):
    eval_id = "abstract_narrative_understanding"
    eval_template = "match"
    subtasks = ["4_distractors", "9_distractors", "99_distractors"]
    sys_msg = "I will provide you a question and multiple choice options. Answer with the correct index."
    ideal_index = True
    append_choices_to_input = True  # it is stored incorrect in the original json, but it definitely needs this.


@dataset_class("cause_and_effect")
class CauseAndEffect(DatasetPreparerBIGBench):
    eval_id = "cause_and_effect"
    eval_template = "match"
    subtasks = ["one_sentence", "one_sentence_no_prompt", "two_sentences", ]
    sys_msg = "I will provide you a question and two options. Answer with the correct index (1 or 2)."
    ideal_index = True


@dataset_class("goal_step_wikihow")
class GoalStepWikihow(DatasetPreparerBIGBench):
    eval_id = "goal_step_wikihow"
    eval_template = "match"
    subtasks = ["goal_inference", "step_inference", "step_ordering"]
    sys_msg = "I will provide you a question about a WikiHow guide and four options. Answer with the correct index (1, 2, 3, or 4)."
    ideal_index = True


@dataset_class("arithmetic")
class Arithmetic(DatasetPreparerBIGBench):
    eval_id = "arithmetic"
    eval_template = "match"
    # subtasks = ["1_digit_addition", "2_digit_addition"]
    subtasks = [[f"{n}_digit_{operation}" for n in range(1, 6)] for operation in
                ["addition", "subtraction", "multiplication", "division"]]
    # create a single list
    subtasks = [item for sublist in subtasks for item in sublist]
    sys_msg = "I will provide you with an arithmetic problem. Answer by only reporting the number that is the right answer."
    ideal_index = False
    append_choices_to_input = False


if __name__ == "__main__":
    # set up arg parser that can list the datasets that you want to set up
    parser = argparse.ArgumentParser(description='Set up datasets')
    parser.add_argument('--datasets', nargs='+',
                        help='list of datasets to set up; if no dataset is provided, all datasets are downloaed',
                        choices=list(DATASETS_DICT.keys()), default=None)

    datasets = parser.parse_args().datasets
    if datasets is None:
        # then prepare all datasets:
        datasets = list(DATASETS_DICT.keys())

    for dataset_name in datasets:
        print(f"Downloading dataset {dataset_name}...")
        dataset_class = DATASETS_DICT[dataset_name]
        dataset = dataset_class()

        raw_data = os.path.join("1_raw_datasets", dataset_name)
        registry = os.path.join("1_evals_registry")
        # create the folders if they do not exist
        os.makedirs(raw_data, exist_ok=True)
        os.makedirs(registry, exist_ok=True)

        dataset.download_raw_data(raw_data)
        print(f"Transforming dataset {dataset_name}...")
        dataset.transform_raw_data(raw_data, registry, rng_seed=42)
        print(f"Dataset {dataset_name} is set up.")
        dataset.register_dataset(registry)
