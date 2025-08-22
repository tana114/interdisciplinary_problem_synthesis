import os

from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

'''
環境変数に`HF_TOKEN`としてHuggingFaceのトークンを登録してください
'''

HF_TOKEN_VARIABLE = "HF_TOKEN"

@dataclass
class HuggingHubConfig:
    """
    dataclassでまとめて取り扱う
    """
    repo_id: str
    config_name: str


def push_to_hub(dataset_dict: DatasetDict, hf_config: HuggingHubConfig):
    load_dotenv()
    api_key = os.getenv(HF_TOKEN_VARIABLE)
    login(api_key)

    repo_id = hf_config.repo_id
    config_name = hf_config.config_name if hf_config.config_name else "default"

    dataset_dict.push_to_hub(
        repo_id=repo_id,
        config_name=config_name
    )


def main():
    import pandas as pd

    hf_cfg_dic = dict(
        repo_id="hoge/math-3_test",
        config_name="sample_data"
    )

    hf_cfg = HuggingHubConfig(**hf_cfg_dic)

    jsonl_path = './data/hogefuga.jsonl'
    dataset = Dataset.from_pandas(pd.read_json(jsonl_path, lines=True))

    dataset_dict = DatasetDict(
        {"train": dataset},
    )

    push_to_hub(dataset_dict, hf_cfg)


if __name__ == "__main__":
    """
    python -m util.hub_push_tools
    """
    main()
