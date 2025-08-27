from dataclasses import dataclass, field
from collections import defaultdict
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Set
import os

from tqdm.auto import tqdm

from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

from manager.sympy_coincidence_counter import CoincidenceCounter
from util.file_tools import JsonHandler, JsonlHandler
from util.hub_push_tools import HuggingHubConfig, push_to_hub

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class HfSympySelfConsistencyConfig:
    """
    huggingface dataset

    "id", "rollout", "answer_sympy", "reasoning"カラムが必要

    """
    output_dir: str
    seed_repo_id: str
    seed_name: Optional[List[str]]
    end_id: Optional[int]
    start_id: int = 1
    seed_split: str = "train"
    output_file_decoration: str = "_"  # 生成したファイルのファイル名に追加する文字列
    batch_size: int = 1  # 生成処理を何個づつまとめて実施して保存するか（基本1で良い）
    seed_data_keys: Set[str] = field(default_factory=lambda: {"id", "rollout", "answer_sympy", "reasoning"})


def group_jsonl_by_id(jsonl_data: List[Dict], group_key: str = "answers") -> List[Dict]:
    """
    from manager.sympy_coincidence_counter import CoincidenceCounter
    フラットなjsonlデータを {"id": x, "answers": [...]} 形式に変換
    """
    grouped = defaultdict(list)
    for entry in jsonl_data:
        # "id" を先頭に置き、それ以外は元のキーをそのまま展開
        grouped[entry["id"]].append({"id": entry["id"], **entry})
    return [{"id": k, group_key: v} for k, v in grouped.items()]


class SympySelfConsistencyManager(object):
    def __init__(
            self,
    ):
        """
        """
        pass


    def __call__(
            self,
            manager_cfg: HfSympySelfConsistencyConfig,
            hf_cfg: Optional[HuggingHubConfig] = None,
    ) -> None:
        self.file_handling(manager_cfg, hf_cfg)

    @staticmethod
    def update_or_append_efficient(original_list, new_list):
        """
        より効率的な方法（大きなリストの場合に有利）
        """
        # 辞書に変換（idをキーとして）
        item_dict = {item["id"]: item for item in original_list}

        # 新しい要素で更新
        for new_item in new_list:
            item_dict[new_item["id"]] = new_item

        # 辞書をリストに戻す
        return list(item_dict.values())

    def file_handling(
            self,
            manager_cfg: HfSympySelfConsistencyConfig,
            hf_cfg: Optional[HuggingHubConfig]
    ):

        out_suffix = '.json'
        output_base_dir = manager_cfg.output_dir
        seed_repo_id = manager_cfg.seed_repo_id
        seed_name = manager_cfg.seed_name
        seed_split = manager_cfg.seed_split
        output_file_decoration = manager_cfg.output_file_decoration

        output_file_name = seed_repo_id + output_file_decoration + out_suffix
        fp_output_file = Path(output_base_dir + output_file_name)

        if not fp_output_file.parent.exists():
            fp_output_file.parent.mkdir(parents=True, exist_ok=True)

        # jsonファイルを読み書きするツール
        jh = JsonHandler()

        # データセットの読み込み
        if seed_name:
            dfs = [load_dataset(seed_repo_id, name=n, split=seed_split).to_pandas() for n in seed_name]
            df = pd.concat(dfs, ignore_index=True)
        else:
            dataset = load_dataset(seed_repo_id, split=seed_split)
            df = dataset.to_pandas()

        # Check required keys
        missing_keys = manager_cfg.seed_data_keys - set(df.columns.to_list())
        if missing_keys:
            error_msg = f"Key required for dictionary data is missing.\n Missing keys: {missing_keys}\n repo_id: {seed_repo_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        start_id = manager_cfg.start_id
        end_id = manager_cfg.end_id

        if end_id:
            df = df[df['id'].between(start_id, end_id, inclusive='both')]
        else:
            df = df[df['id'].between(start_id, df['id'].max(), inclusive='both')]

        if os.path.isfile(fp_output_file):
            gen_objects = jh.read(str(fp_output_file))
            ''' 処理を効率化するためにidをkeyにした辞書型に変換。↓これでもとに戻る
            gen_objects = list(existing_results.values())
            '''
            existing_results = {item['id']: item for item in gen_objects}  # id:items
        else:
            gen_objects = []
            existing_results = {}

        # print(existing_results)

        ''' pandas_dataframeを辞書型に変換し、さらに"id"keyで纏めた形に変換'''
        id_key_df = group_jsonl_by_id(df.to_dict('records'), group_key="answers")

        counter = CoincidenceCounter(id_key_df, group_key="answers", sympy_key="answer_sympy",
                                     leading_target_key="reasoning")

        consistent_data = []
        for i in tqdm(range(start_id, end_id + 1)):
            consistent_data = counter.update_base_json(target_id=i)
            # print(consistent_data [0].keys())  # dict_keys(['id', 'answers', 'rollout', 'question', 'answer', 'output', 'reasoning', 'answer_sympy', 'num_of_coincidences', 'num_of_rollout', 'consistent_rate']
            # print(consistent_data)

            # ファイルから読み込んだデータに上書き
            for add_item in consistent_data:
                item_id = add_item['id']
                # print(add_item)
                if add_item and len(add_item.keys()) != 1:
                    existing_results[item_id] = add_item

            # print(existing_results)
            gen_objects = list(existing_results.values())
            # print(gen_objects)
            jh.write(gen_objects, str(fp_output_file))

        # huggingfaceにpush
        if hf_cfg is not None and hf_cfg.repo_id:

            ''' rollout_data
             rolloutした全てのanswer '''
            rollout_answers = [answer for item in gen_objects for answer in item["answers"]]
            df_rollout = pd.DataFrame(rollout_answers)
            # print(df_rollout.head())
            dataset = Dataset.from_pandas(df_rollout)
            dataset_dict = DatasetDict(
                {"train": dataset},
            )
            rollout_hf_cfg = HuggingHubConfig(hf_cfg.repo_id, hf_cfg.config_name + "_rollout")
            push_to_hub(dataset_dict, rollout_hf_cfg)

            ''' leading_data
             self consistency の値が最も高い回答をピックしたもの '''
            # keys_to_exclude = {'answers', 'rollout'}
            keys_to_exclude = {'answers',}  # 'answers'は除外
            leading_objects = [{k: v for k, v in d.items() if k not in keys_to_exclude} for d in gen_objects]
            df_leading = pd.DataFrame(leading_objects)
            # 'rollout'を一番右に移動
            cols = [col for col in df_leading.columns if col != 'rollout'] + ['rollout']
            df_leading = df_leading[cols]
            # print(df_leading.head())
            dataset = Dataset.from_pandas(df_leading)
            dataset_dict = DatasetDict(
                {"train": dataset},
            )
            push_to_hub(dataset_dict, hf_cfg)




if __name__ == "__main__":
    """
    python -m manager.sympy_self_consistency_manager
    """

    # def fix_seed(seed):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #
    #
    # SEED = 46
    # fix_seed(SEED)

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)
    # basicConfig(level=DEBUG)

    rom = SympySelfConsistencyManager()

    test_cfg = dict(
        output_dir='./data/consistency/',
        seed_repo_id="tarona/MathXPhys_scored_v1",
        seed_name=["OB_PHYS_rollout_sympy", ],
        output_file_decoration="-v1.0_self_consistency",
        start_id=1,
        end_id=200,
    )

    data_config = HfSympySelfConsistencyConfig(**test_cfg)
    # rom(data_config)

    # huggingfaceの設定ファイルを作成（値を指定していない場合はpushしない）
    hf_repo_id = "tarona/MathXPhys_scored_v1"
    hf_config_name = "OB_PHYS_self_consistency"
    hf_cfg = HuggingHubConfig(repo_id=hf_repo_id, config_name=hf_config_name)

    rom(data_config, hf_cfg)
