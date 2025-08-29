from dataclasses import dataclass, field

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Set
import os
import time

from tqdm.auto import tqdm

from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

from client.concrete.simple_answer_gen import SimpleAnswerGenerator

from util.file_tools import JsonHandler, JsonlHandler
from util.hub_push_tools import HuggingHubConfig, push_to_hub

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class HfRolloutConfig:
    """
    Hugging Face dataset.

    The dataset must contain the columns: "id", "question", and "final_score".
    "final_score" is a numeric quality score that measures the quality of the synthesized "question" text.

    """
    output_dir: str
    seed_repo_id: str
    seed_name: Optional[List[str]]
    end_id: Optional[int]
    start_id: int = 1
    seed_split: str = "train"
    output_file_decoration: str = "_"  # String appended to generated filenames
    batch_size: int = 1  # Number of items to process per generation/save batch (default 1)
    final_score_threshold: float = 7.0  # Only items with "final_score" above this threshold are considered for rollout
    seed_data_keys: Set[str] = field(default_factory=lambda: {"id", "question", "final_score"})


def extract_last_boxed(text: str) -> str | None:
    last = None
    i = 0
    while True:
        start = text.find(r'boxed{', i)
        if start == -1:
            break
        i = start + len(r'boxed{')
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            buf.append(text[i])
            i += 1
        last = ''.join(buf)
    return last


class RolloutManager(object):
    def __init__(
            self,
            model_name: str,
            num_of_rollout: int = 10,
    ):
        """
        """
        self._model = model_name
        self._num_of_rollout = num_of_rollout

    def __call__(
            self,
            manager_cfg: HfRolloutConfig,
            hf_cfg: Optional[HuggingHubConfig] = None,
    ) -> None:
        self.file_handling(manager_cfg, hf_cfg)

    def answer_gen(self, task_seeds: List[Dict]) -> List[Dict]:

        conversations = []

        for seed in task_seeds:
            seed_id = seed['id']
            seed_question = seed['question']

            instruction = {
                "question": str(seed_question),
            }

            conversations = []

            answer_gen = SimpleAnswerGenerator(self._model)

            message = answer_gen.message(
                instruction,
                temperature=0.6,
                top_p=0.95,
            )

            if message:
                msg = message.to_dict()
                output = msg.get('content', '')
                reasoning = msg.get('reasoning', '')
                answer = extract_last_boxed(str(output))

                gen_data = {
                    "id": seed_id,
                    "answer": answer,
                    "output": output,
                    "reasoning": reasoning
                }
            else:
                gen_data = {
                    "id": seed_id,
                    "answer": None,
                    "output": None,
                    "reasoning": None
                }

            # '''For testing / sanity check'''
            # sympy_ans = random.sample(["a**2+2*a*b+b**2", "(a+b)**2", "3", "c**2+4,(a+b)**3"], 1)
            # ans_str = str(sympy_ans[0])
            #
            # gen_data = {
            #     "id": seed_id,
            #     "answer": ans_str,
            #     "output": "hoge",
            #     "reasoning": "fuga"
            # }

            conversations.append(gen_data)

        return conversations

    def file_handling(
            self,
            manager_cfg: HfRolloutConfig,
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

        # Tool for reading/writing JSON files
        jh = JsonHandler()

        # Load dataset
        if seed_name:
            dfs = [load_dataset(seed_repo_id, name=n, split=seed_split).to_pandas() for n in seed_name]
            df = pd.concat(dfs, ignore_index=True)
        else:
            dataset = load_dataset(seed_repo_id, split=seed_split)
            df = dataset.to_pandas()

        # Check required keys
        missing_keys = manager_cfg.seed_data_keys - set(df.columns.to_list())
        if missing_keys:
            error_msg = f"Key required for dictionary data is missing.\n Missing keys: {missing_keys}\n repo_id: {cfg.seed_repo_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        start_id = manager_cfg.start_id
        end_id = manager_cfg.end_id

        if end_id:
            df = df[df['id'].between(start_id, end_id, inclusive='both')]
        else:
            df = df[df['id'].between(start_id, df['id'].max(), inclusive='both')]

        # Filter rows where final_score > score_threshold
        score_threshold = manager_cfg.final_score_threshold
        df = df[df['final_score'] > score_threshold]

        if os.path.isfile(fp_output_file):
            gen_objects = jh.read(str(fp_output_file))
            ''' For efficiency, convert to a dict keyed by 'id'.
            You can convert back with:
            gen_objects = list(existing_results.values())
            '''
            existing_results = {item['id']: item for item in gen_objects}  # id:items
        else:
            # gen_objects = {}
            existing_results = {}

        # Prepare results dict
        results_dict = existing_results.copy()

        # For each id, generate rollout answers
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            seed_id = row['id']
            seed_dict = row.to_dict()  # "id", "question"
            current_item = results_dict.get(seed_id, {'id': seed_id})
            current_item['question'] = seed_dict.get('question', '')
            results_dict[seed_id] = current_item

            answers = current_item.get('answers', [])  # previously rolled out answers
            current_count = len(answers)

            ''' 1. Perform rollouts up to the specified count '''
            # Initialize progress bar
            pbar = tqdm(
                total=self._num_of_rollout,
                initial=current_count,
                desc=f"Rollout Progress",
            )
            while current_count < self._num_of_rollout:

                def batch_processor(data, b_size):
                    yield self.answer_gen([data] * b_size)

                for processed in tqdm(batch_processor(seed_dict, manager_cfg.batch_size)):
                    if processed:
                        answers.extend(processed)
                        item = results_dict[seed_id]
                        # Add rollout index/number
                        answers = [{**d, "rollout": i} for i, d in enumerate(answers, start=1)]
                        item['answers'] = answers
                        results_dict[seed_id] = item
                        gen_objects = list(results_dict.values())
                        jh.write(gen_objects, str(fp_output_file))

                new_count = len(answers)

                pbar.update(new_count - current_count)
                current_count = new_count

                if current_count >= self._num_of_rollout:
                    break

            # Finalize/complete the progress bar
            gen_objects = list(results_dict.values())
            jh.write(gen_objects, str(fp_output_file))
            pbar.close()

        ''' 2. Convert to a flat format and push to Hugging Face (HF) '''
        if hf_cfg is not None and hf_cfg.repo_id:
            # Convert to a flat structure
            flattened_data = []
            for item in gen_objects:
                for answer in item['answers']:
                    flattened_data.append({
                        'id': item['id'],
                        'rollout': answer['rollout'],
                        'question': item['question'],
                        'answer': answer['answer'],
                        'output': answer['output'],
                        'reasoning': answer['reasoning'],
                    })

            # Create a Dataset
            dataset = Dataset.from_list(flattened_data)
            dataset_dict = DatasetDict(
                {"train": dataset},
            )
            push_to_hub(dataset_dict, hf_cfg)


if __name__ == "__main__":
    """
    python -m manager.rollout_manager
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    model_name = 'deepseek/deepseek-r1-0528:free'

    rom = RolloutManager(model_name, num_of_rollout=10)

    test_cfg = dict(
        output_dir='./data/rollout/',
        seed_repo_id="tarona/MathXPhys_scored_v1",
        seed_name=["OB_PHYS_problem", ],
        output_file_decoration="-v1.0_rollout",
        start_id=201,
        end_id=300,
    )

    data_config = HfRolloutConfig(**test_cfg)
    rom(data_config)

    # Create Huggingface config file (It will not push if the config are not specified)
    hf_repo_id = "tarona/MathXPhys_scored_v1"
    hf_config_name = "OB_PHYS_rollout"
    hf_cfg = HuggingHubConfig(repo_id=hf_repo_id, config_name=hf_config_name)

    rom(data_config, hf_cfg)
