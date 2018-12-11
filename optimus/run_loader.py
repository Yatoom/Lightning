import json
import re

import openml
import pandas as pd
import requests
from tqdm import tqdm

class RunLoader:
    @staticmethod
    def build_query(task_id, flow_id):
        return {
            "_source": [
                "run_id",
                "date",
                "run_flow.name",
                "run_flow.flow_id",
                "evaluations.evaluation_measure",
                "evaluations.value",
                "run_flow.parameters.parameter",
                "run_flow.parameters.value",
                "run_task.source_data.data_id"
            ],
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "run_task.task_id": task_id
                            },
                        },
                        {
                            "term": {
                                "run_flow.flow_id": flow_id
                            },
                        },
                        {
                            "nested": {
                                "path": "evaluations",
                                "query": {
                                    "exists": {
                                        "field": "evaluations"
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "sort": {
                "date": "asc"
            }
        }

    @staticmethod
    def load_task_runs(task_id, flow_id, metric, size=5000):
        query = RunLoader.build_query(task_id, flow_id)
        url = f"https://www.openml.org/es/run/run/_search?size={size}"
        result = json.loads(requests.request(method="post", url=url, json=query).content)

        if len(result["hits"]["hits"]) == 0:
            return None, []

        # Get parameter names
        columns = [i["parameter"] for i in result["hits"]["hits"][0]["_source"]["run_flow"]["parameters"]]
        converted = [RunLoader.convert_param_name(i) for i in columns]

        # Construct header
        header = ["data_id", "task_id", "run_id", metric, *converted]

        # Get all values
        values = [
            # Dataset id
            [int(result["hits"]["hits"][0]["_source"]["run_task"]["source_data"]["data_id"])] +

            # Task id
            [int(task_id)] +

            # Run id
            [int(sample["_source"]["run_id"])] +

            # Metric
            [
                [i["value"] for i in sample["_source"]["evaluations"] if i["evaluation_measure"] == metric]
                or [None]
            ][0] +

            # Parameter values
            [json.loads(i["value"]) for i in sample["_source"]["run_flow"]["parameters"]]

            for sample in result["hits"]["hits"]
        ]

        return header, values

    @staticmethod
    def load_tasks(tasks, flow_id, metric="predictive_accuracy", max_per_task=5000):
        all = []
        columns = None
        for task in tqdm(tasks):
            header, values = RunLoader.load_task_runs(task_id=task, flow_id=flow_id, metric=metric, size=max_per_task)
            if values:
                columns = header
                all += values
        return pd.DataFrame(all, columns=columns)

    @staticmethod
    def convert_param_name(param_name):
        """
        Examples:
        sklearn.feature_selection.variance_threshold.VarianceThreshold(4)_threshold
        --> variancethreshold__threshold

        (...).AdaBoostClassifier(base_estimator=(...).DecisionTreeClassifier)(2)_random_state
        --> decisiontreeclassifier__random_state

        :param param_name: Parameter name to convert
        :return: (str) Converted parameter name

        """
        splits = re.compile(r"(?:\(.*\))+_").split(param_name)
        prefix = splits[0].split(".")[-1].lower()
        postfix = splits[1]
        result = f"{prefix}__{postfix}"
        return result

    @staticmethod
    def get_cc18_benchmarking_suite():
        return [146825, 146800, 146822, 146824, 167119, 146817, 14954, 37, 219, 9964, 3573, 12, 9957, 14970, 9946, 31, 3021,
                146195, 18, 29, 11, 53, 6, 23, 2079, 14969, 3918, 3902, 9976, 15, 16, 32, 125922, 167120, 167121, 167124,
                167125, 146819, 167141, 9910, 14952, 146818, 167140, 146820, 9952, 3904, 14, 49, 2074, 3022, 3481, 43, 3903,
                9971, 3, 28, 9978, 7592, 3549, 22, 9985, 9960, 3913, 9977, 3560, 10101, 45, 10093, 146821, 3917, 9981,
                125920, 14965]