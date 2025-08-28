from dataclasses import dataclass, field
from collections import defaultdict
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Set
import os

from tqdm.auto import tqdm

from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


from client.concrete.tex2sympy_gen import SympyConversionGenerator

from util.file_tools import JsonHandler, JsonlHandler
from util.hub_push_tools import HuggingHubConfig, push_to_hub

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class HfSympyConfig:
    """
    Hugging Face dataset.

    Columns "id", "rollout", and "answer" are required.

    """
    output_dir: str
    seed_repo_id: str
    seed_name: Optional[List[str]]
    end_id: Optional[int]
    start_id: int = 1
    seed_split: str = "train"
    output_file_decoration: str = "_"  # string appended to generated file names
    batch_size: int = 1  # how many items to generate and save at once (1 is usually fine)
    seed_data_keys: Set[str] = field(default_factory=lambda: {"id", "rollout", "answer" })




class SympyConversionManager(object):
    def __init__(
            self,
            model_name:str,
    ):
        """
        """
        self._model = model_name

    def __call__(
            self,
            manager_cfg: HfSympyConfig,
            hf_cfg: Optional[HuggingHubConfig] = None,
    ) -> None:
        self.file_handling(manager_cfg, hf_cfg)

    def answer_gen(self, task_seeds: List[Dict]) -> List[Dict]:

        conversations = []

        for seed in task_seeds:
            # seed_id = seed['id']
            seed_answer = seed['answer']

            instruction = {
                "tex_text": str(seed_answer),
            }

            conversations = []
            
            answer_gen = SympyConversionGenerator(self._model)

            parse = answer_gen.parse(
                instruction,
                temperature=0.1,
                top_p=0.95,
            )

            if parse:
                sympy_answer = parse.get('Sympy_text', '')
                # msg = message.to_dict()
                # output = msg.get('content', '')
                # reasoning = msg.get('reasoning', '')
                # answer =  extract_last_boxed(str(output))

                gen_data = {
                    **seed,
                    "answer_sympy": sympy_answer,
                }
            else:
                gen_data = {
                    **seed,
                    "answer_sympy": None,
                }

            conversations.append(gen_data)

        return conversations

    @staticmethod
    def update_or_append_efficient(original_list, new_list):
        # 辞書に変換（idをキーとして）
        item_dict = {item["id"]: item for item in original_list}

        # 新しい要素で更新
        for new_item in new_list:
            item_dict[new_item["id"]] = new_item

        # 辞書をリストに戻す
        return list(item_dict.values())

    def file_handling(
            self,
            manager_cfg: HfSympyConfig,
            hf_cfg: Optional[HuggingHubConfig]
    ):

        out_suffix = '.jsonl'
        output_base_dir = manager_cfg.output_dir
        seed_repo_id = manager_cfg.seed_repo_id
        seed_name = manager_cfg.seed_name
        seed_split = manager_cfg.seed_split
        output_file_decoration = manager_cfg.output_file_decoration

        output_file_name = seed_repo_id + output_file_decoration + out_suffix
        fp_output_file = Path(output_base_dir + output_file_name)

        if not fp_output_file.parent.exists():
            fp_output_file.parent.mkdir(parents=True, exist_ok=True)

        # tool for reading/writing jsonl files
        jlh = JsonlHandler()

        # Load dataset
        # rm -r ~/.cache/huggingface/datasets/tarona___math_x_phys_scored_v1
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


        ''' To speed up processing, convert to a dict keyed by id. It can be converted back like this:
        def flatten_nested_dict(nested_dict):
            result = []
            for id_dict in nested_dict.values():
                for item in id_dict.values():
                    result.append(item)
            return result
        '''
        existing_results = defaultdict(dict)
        if os.path.isfile(fp_output_file):
            gen_objects = jlh.read(str(fp_output_file))
            for item in gen_objects:
                existing_results[item['id']][item['rollout']] = item


        def batch_processor(data, b_size):
            for i in range(0, len(data), b_size):
                yield self.answer_gen(data[i:i + b_size])

        for processed in tqdm(batch_processor(df.to_dict('records'), manager_cfg.batch_size)):
            for new_item in processed:
                existing_results[new_item["id"]][new_item["rollout"]] = new_item

            result = []
            for id_dict in existing_results.values():
                for item in id_dict.values():
                    result.append(item)
            jlh.write(result, str(fp_output_file))

        # push to Hugging Face
        if hf_cfg is not None and hf_cfg.repo_id:
            dataset = Dataset.from_pandas(pd.read_json(fp_output_file, lines=True))
            dataset_dict = DatasetDict(
                {"train": dataset},
            )
            push_to_hub(dataset_dict, hf_cfg)


if __name__ == "__main__":
    """
    python -m manager.sympy_conversion_manager
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

    model_name = "Qwen/Qwen3-8B-AWQ"

    rom = SympyConversionManager(model_name)

    test_cfg = dict(
        output_dir='./data/rollout/',
        seed_repo_id="tarona/MathXPhys_scored_v1",
        seed_name=["OB_PHYS_rollout", ],
        output_file_decoration="-v1.0_rollout_sympy",
        start_id=101,
        end_id=200,
    )
    
    data_config = HfSympyConfig(**test_cfg)
    # rom(data_config)

    # Create Huggingface config file (It will not push if the config are not specified)
    hf_repo_id = "tarona/MathXPhys_scored_v1"
    hf_config_name ="OB_PHYS_rollout_sympy"
    hf_cfg = HuggingHubConfig(repo_id=hf_repo_id, config_name=hf_config_name)

    rom(data_config, hf_cfg)
