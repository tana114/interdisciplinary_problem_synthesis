
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic

from pydantic import BaseModel, Field

from client.concrete.prompt_loader import load_prompt_config
from client.client_base_local import ApiClientBase

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

prompt_config = load_prompt_config(version='1.0', prompt_type="tex2sympy")
SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']


class SympyDate(BaseModel):
    """ Formulas converted to sympy format. """
    Sympy_text: str = Field(description="Formulas converted to Python sympy format.")

class SympyConversionGenerator(ApiClientBase):

    def _create_message_config(self, prompt_elements: Dict):
        system_prompt = SYSTEM_PROMPT_FORMAT
        # prompt_elements -> Dict[Literal["tex_text"], str],
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
        system_prompt = SYSTEM_PROMPT_FORMAT
        # prompt_elements -> Dict[Literal["tex_text"], str],
        user_prompt = USER_PROMPT_FORMAT.format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model_name,
            "messages": messages,
            "response_format": SympyDate,
        }

        return config

    

if __name__ == "__main__":
    """
    python -m client.concrete.tex2sympy_gen
    """

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

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    question = "\left( \frac{m_{\mu} (m_{e} + A m_{p})}{m_{e} (m_{\mu} + A m_{p})} \right)^{3}"

    gen = SympyConversionGenerator(
        model_name="Qwen/Qwen3-8B-AWQ",
    )

    inst = {
        "tex_text": question,
    }

    # content = gen.content(
    #     inst,
    #     temperature=0.1,
    #     top_p=0.95,
    # )
    #
    # print(type(content ))
    # print(content)

    parse = gen.parse(
        inst,
        temperature=0.1,
        top_p=0.95,
    )

    print(type(parse))
    print(parse)

