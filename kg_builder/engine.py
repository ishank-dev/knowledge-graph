import copy
import os
from abc import ABC, abstractmethod
from functools import cached_property

import torch
from google import genai
from google.genai.types import GenerateContentConfig
from tenacity import retry, wait_exponential, stop_after_attempt
from transformers import BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel
from transformers import DynamicCache, AutoTokenizer, AutoModelForCausalLM

from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting


class ChatEngine(ABC):
    def __init__(self, model_id: str, prompting: ExtractorPrompting | BuilderPrompting, **kwargs):
        self.model_id = model_id
        self.prompting = prompting

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def clear_caches(self):
        raise NotImplementedError


class CachedInstructionsEngine(ChatEngine):
    def __init__(self,
                 model_id: str,
                 prompting: ExtractorPrompting | BuilderPrompting,
                 quantization_config: BitsAndBytesConfig | None = None,
                 **model_kwargs):
        super().__init__(model_id=model_id, prompting=prompting)
        self.model_kwargs = model_kwargs
        self.quantization_config = quantization_config
        self.quantization_config = quantization_config

    def clear_caches(self):
        del self.tokenizer
        del self.model
        del self.system_prompt_kv_cache
        torch.cuda.empty_cache()

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.model_id)

    @cached_property
    def model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(self.model_id,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda",
                                                    trust_remote_code=True,
                                                    quantization_config=self.quantization_config,
                                                    **self.model_kwargs)

    @cached_property
    def system_prompt_kv_cache(self) -> DynamicCache:
        message = [{"role": "system", "content": self.prompting.system_prompt}]
        system_msg = self.tokenizer.apply_chat_template(message,
                                                        add_generation_prompt=False,
                                                        return_tensors="pt",
                                                        return_dict=True,
                                                        ).to(self.model.device)
        with torch.no_grad():
            prompt_cache = self.model(**system_msg).past_key_values
        return prompt_cache

    def chat_completion(self,
                        messages: list[dict[str, str]],
                        max_new_tokens: int = 4096,
                        **kwargs) -> str:
        inputs = self.tokenizer.apply_chat_template(messages,
                                                    return_tensors="pt",
                                                    return_dict=True,
                                                    add_generation_prompt=True).to(self.model.device)
        kv_cache = copy.deepcopy(self.system_prompt_kv_cache)
        generated = self.model.generate(**inputs,
                                        max_new_tokens=max_new_tokens,
                                        past_key_values=kv_cache,
                                        cache_implementation=None,
                                        **kwargs)
        decoded = self.tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:])
        return decoded


class GeminiEngine(ChatEngine):
    def __init__(self,
                 model_id: str,
                 prompting: BuilderPrompting | ExtractorPrompting):
        super().__init__(model_id=model_id, prompting=prompting)

    @cached_property
    def client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY", None)
        if api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use Gemini model.")
        return genai.Client(api_key=api_key)

    @retry(wait=wait_exponential(multiplier=2, min=10, max=100), stop=stop_after_attempt(5))
    def chat_completion(self, messages: list[dict[str, str]], temperature: float = 0., **kwargs) -> str:
        chat = self.client.chats.create(model=self.model_id)
        cfg = GenerateContentConfig(system_instruction=self.prompting.system_prompt,
                                    temperature=temperature)
        response = chat.send_message(messages[-1]["content"], config=cfg)
        return response.text

    def clear_caches(self):
        del self.client
