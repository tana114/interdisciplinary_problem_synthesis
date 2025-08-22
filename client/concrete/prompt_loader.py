
import yaml
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from pathlib import Path

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt_config(version: str, prompt_type: str) -> Dict[str, Any]:
    """プロンプト設定をYAMLから読み込む
    以下より引用させていただきました
    https://github.com/Damin3927/llm2025compet/tree/feat/sft-physics-pipeline
    """
    if not version:
        raise ValueError("Prompt version must be specified")

    # レジストリを読み込み
    registry_path = PROMPTS_DIR / "registry.yaml"
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)

    # version_infoをversionとtypeの両方で取得
    version_info = next(
        (v for v in registry['versions'] if v['version'] == version and v.get('type', 'generate') == prompt_type), None
    )

    if not version_info:
        # 利用可能なバージョンとタイプの組み合わせを表示
        available = [(v['version'], v.get('type', 'generate')) for v in registry['versions']]
        raise ValueError(f"Version {version} with type {prompt_type} not found. Available: {available}")

    # プロンプトファイルを読み込み
    with open(PROMPTS_DIR / version_info['file'], 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"\nLoaded prompt version: {version}")
    logger.info(f"  Status: {version_info['status']}")
    logger.info(f"  Summary: {version_info['description']}")
    return config


''' usage
prompt_config = load_prompt_config(version='2.2', prompt_type="math_phys_easy_generate")
SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']
'''