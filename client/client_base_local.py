import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic, Tuple, cast
from abc import ABCMeta, abstractmethod
import time

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

import json

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

'''
1.ローカルでvllmサーバーを起動させた状態で実行すること
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-AWQ \
--max-model-len 20480 --port 8000 \
--gpu-memory-utilization 0.75 --trust-remote-code \
--tensor-parallel-size 1 --api-key 1234 \
--swap-space 16

＊ローカルで起動させたモデルとクラスで指定したモデル名が一致していないとエラー

# 別ターミナルで停止 (Ctrl+Cでも可)
pkill -f "vllm.entrypoints.api_server"
'''


API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "1234"


class ApiClientBase(metaclass=ABCMeta):

    ''' Variable for checking the number of requests '''
    message_counter:int = 0
    parse_counter:int = 0

    def __init__(
            self,
            model_name: str,
            api_key: Optional[str] = API_KEY ,
    ):
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
        )
        self._model_name = model_name


    @abstractmethod
    def _create_message_config(self, prompt_elements: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def _create_parse_config(self, prompt_elements: Dict) -> Dict:
        raise NotImplementedError

    def message(
            self,
            prompt_elements: Dict,
            **kwargs,
    ) -> Optional[ChatCompletionMessage]:
        try:

            config: Dict = self._create_message_config(prompt_elements)
            merged_dict = config | kwargs
            
            ''' ------------------------------------ '''
            ApiClientBase.message_counter += 1
            logger.info(f"message requests : {ApiClientBase.message_counter}")
            ''' ------------------------------------ '''
            response = self._client.chat.completions.create(**merged_dict)
            if response:
                return response.choices[0].message
            else:
                return None

        except openai.RateLimitError as e:
            delay = 10
            logger.warning(f"Too Many Requests : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)

        except json.JSONDecodeError as e:
            logger.warning(f" JSONDEcodeError during API request: {e} ")
            return None
        

    def content(
            self,
            prompt_elements: Dict,
            **kwargs,
    ) -> Optional[str]:
        response: Optional[ChatCompletionMessage] = self.message(prompt_elements, **kwargs)
        if response:
            return response.content
        else:
            return None

    def parse(
            self,
            prompt_elements: Dict,
            **kwargs
    ) -> Optional[Dict]:
        try:
            config: Dict = self._create_parse_config(prompt_elements)
            merged_dict = config | kwargs
            
            ''' ------------------------------------ '''
            ApiClientBase.parse_counter += 1
            logger.info(f"parse requests : {ApiClientBase.parse_counter}")
            ''' ------------------------------------ '''
            # print(merged_dict)
            response = self._client.chat.completions.parse(**merged_dict)

            if response:
                content = cast(BaseModel, response.choices[0].message.parsed) 
                return content.model_dump()
            else:
                return None

        except openai.RateLimitError as e:
            delay = 10
            logger.warning(f"Too Many Requests : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)

        except json.JSONDecodeError as e:
            logger.warning(f" JSONDEcodeError during API request: {e} ")
            return None

        except ValidationError as e:
            logger.warning(f"pydantic ValidationError occurred in the API request: {e} ")
            return None

''' ----------- simple example ------------ '''

class SolutionDate(BaseModel):
    """ Final answer to the question. """
    Solution: str = Field(description="Solution to the questions.")
    Final_Answer: str = Field(description="the final answer(s).")


class SympleApiClient(ApiClientBase):

    def _create_message_config(self, prompt_elements: Dict) -> Dict:
        system_prompt = "You are helpfull assistant."
        user_prompt = "solve this ploblem: {question}".format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model_name,
            "messages": messages,
        }

        return config

    def _create_parse_config(self, prompt_elements: Dict) -> Dict:
        system_prompt = "You are helpfull assistant."
        user_prompt = "solve this ploblem: {question}\n\nPlease output the #Solution# and #Final_Answer#.".format(
            **prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model_name,
            "messages": messages,
            "response_format": SolutionDate,
        }

        return config


if __name__ == "__main__":
    """
    python -m client.client_base_local
    """

    """
        1.ローカルでvllmサーバーを起動させた状態で実行すること
        python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-AWQ \
        --max-model-len 20480 --port 8000 \
        --gpu-memory-utilization 0.75 --trust-remote-code \
        --tensor-parallel-size 1 --api-key 1234 \
        --swap-space 16

        ＊ローカルで起動させたモデルとクラスで指定したモデル名が一致していないとエラー

        # 別ターミナルで停止 (Ctrl+Cでも可)
        pkill -f "vllm.entrypoints.api_server"

    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    client = SympleApiClient(model_name="Qwen/Qwen3-8B-AWQ")

    question = "What is 1234 * 5678?"

    inst = {
        "question": question,
    }

    # message = client.message(
    #     inst,
    #     temperature=0.6,
    #     top_p=0.95,
    # )

    # print(type(message))
    # print(message)
    # print(message.reasoning)
    # print('*' * 60)
    # print(message.content)

    parse = client.parse(
        inst,
        temperature=0.6,
        top_p=0.95,
    )

    print(type(parse))
    print(parse)
