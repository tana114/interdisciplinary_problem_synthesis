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

API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"


class ApiClientBase(metaclass=ABCMeta):

    ''' Variable for checking the number of requests '''
    message_counter:int = 0
    parse_counter:int = 0

    def __init__(
            self,
            model_name: str,
            api_key: Optional[str] = None,
    ):
        self._client = openai.OpenAI(
            api_key=self.__get_apikey(api_key),
            base_url=API_BASE_URL,
        )
        self._model_name = model_name

    @staticmethod
    def __get_apikey(api_key: Optional[str] = None) -> Optional[str]:
        if api_key:
            return api_key
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            return os.getenv(API_KEY_VARIABLE)

    @abstractmethod
    def _create_message_config(self, prompt_elements: Dict) -> Dict:
        """
        system_prompt = SYSTEM_PROMPT_FORMAT
        user_prompt = USER_PROMPT_FORMAT.format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model,
            "messages": messages,
        }

        """
        raise NotImplementedError
        # pass

    @abstractmethod
    def _create_parse_config(self, prompt_elements: Dict) -> Dict:
        """
        system_prompt = SYSTEM_PROMPT_FORMAT
        user_prompt = USER_PROMPT_FORMAT.format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model,
            "messages": messages,
            "response_format": SolutionDate,
        }
        """
        raise NotImplementedError
        # pass

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

            if response and hasattr(response, 'choices'):
                return response.choices[0].message
            else:
                return None

        except openai.RateLimitError as e:
            delay = 30
            logger.warning(f"Too Many Requests : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)
            ApiClientBase.message_counter -= 1
            return self.message(prompt_elements, **kwargs)

        except openai.InternalServerError as e:
            delay = 300
            logger.warning(f"InternalServerError : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)
            ApiClientBase.message_counter -= 1
            return self.message(prompt_elements, **kwargs)

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

            if response and hasattr(response, 'choices'):
                content = cast(BaseModel, response.choices[0].message.parsed) 
                return content.model_dump()
            else:
                return None

        except openai.RateLimitError as e:
            delay = 30
            logger.warning(f"Too Many Requests : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)
            ApiClientBase.parse_counter -= 1
            return self.parse(prompt_elements, **kwargs)

        except openai.InternalServerError as e:
            delay = 300
            logger.warning(f"InternalServerError : {e}.\n Retrying in {delay} seconds...")
            time.sleep(delay)
            ApiClientBase.parse_counter -= 1
            return self.parse(prompt_elements, **kwargs)

        except json.JSONDecodeError as e:
            logger.warning(f" JSONDEcodeError during API request: {e} ")
            return None

        except ValidationError as e:
            logger.warning(f"pydantic ValidationError occurred in the API request: {e} ")
            return None

''' ----------- A Simple Example of Using ApiClientBase ------------ '''

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
    python -m client.client_base
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    question = "What is 1234 * 5678?"

    inst = {
        "question": question,
    }

    # client = SympleApiClient(model_name="deepseek/deepseek-r1-0528:free")
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

    '''
    In December 2025, we confirmed that the “free” version no longer supports Perse processing.
    '''
    client = SympleApiClient(model_name="deepseek/deepseek-r1-0528")
    parse = client.parse(
        inst,
        temperature=0.6,
        top_p=0.95,
    )
    print(type(parse))
    print(parse)
