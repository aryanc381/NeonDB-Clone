# llms/sarvam_llm.py

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field
import requests

class SarvamChat(BaseChatModel, BaseModel):
    api_key: str = Field(...)
    model_name: str = Field(default="sarvam-m")
    endpoint: str = Field(default="https://api.sarvam.ai/v1/chat/completions")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user" if m.type == "human" else m.type, "content": m.content}
                for m in messages
            ],
        }
        headers = {
            "api-subscription-key": self.api_key
        }

        response = requests.post(self.endpoint, headers=headers, json=payload)
        response_json = response.json()

        content = response_json["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        return "sarvam-chat"