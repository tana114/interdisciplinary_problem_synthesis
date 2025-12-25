from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic

from pydantic import BaseModel, Field
from pathlib import Path

# from client.concrete.prompt_loader import load_prompt_config
from client.client_base import ApiClientBase
from util.file_tools_gen import PromptConfigAnalyzer

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

PROMPTS_DIR = Path(__file__).parent / "prompts"
prompt_analyzer = PromptConfigAnalyzer(PROMPTS_DIR)
prompt_config = prompt_analyzer.load_config(version='1.0', prompt_type="simple_answer_v1")

SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']

# prompt_config = load_prompt_config(version='1.0', prompt_type="simple_answer_v1")
# SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
# USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']

class SimpleAnswerGenerator(ApiClientBase):

    def _create_message_config(self, prompt_elements: Dict):
        system_prompt = SYSTEM_PROMPT_FORMAT
        # prompt_elements -> Dict[Literal["question"], str],
        user_prompt = USER_PROMPT_FORMAT.format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model_name,
            "messages": messages,
        }

        return config

    def _create_parse_config(self, prompt_elements: Dict):
        pass


if __name__ == "__main__":
    """
    python -m client.concrete.simple_answer_gen
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    question = "What is 1234 * 5678?"

    gen = SimpleAnswerGenerator(
        # model_name="deepseek/deepseek-r1:free",
        model_name="deepseek/deepseek-r1-0528:free",
        # model_name="deepseek/deepseek-chat-v3-0324:free",
        # model_name="deepseek/deepseek-r1-0528",
        # model_name="deepseek/deepseek-chat-v3-0324",
    )

    inst = {
        "question": question,
    }

    message = gen.message(
        inst,
        temperature=0.6,
        top_p=0.95,
    )

    print(type(message))
    print(message)
    if message:
        print(message.reasoning)
        print('*' * 60)
        print(message.content)
