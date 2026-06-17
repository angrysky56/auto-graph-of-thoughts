"""
This module implements the OpenRouter language model provider interface.
It allows querying models hosted on OpenRouter.ai.
"""

import asyncio
import os
import random
import time
from typing import Dict, List, Union

import backoff
from openai import AsyncOpenAI, OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class OpenRouter(AbstractLanguageModel):
    """
    Handles interactions with OpenRouter models using the provided configuration.

    Inherits from AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "openrouter", cache: bool = False
    ) -> None:
        """
        Initialize the OpenRouter instance with configuration, model details, and caching options.
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config.get(model_name, {})
        # The model_id is the id of the model that is used for openrouter,
        # e.g. meta-llama/llama-3-70b-instruct, etc.
        self.model_id: str = os.getenv(
            "OPENROUTER_MODEL_ID", self.config.get("model_id", "openrouter/auto")
        )
        # Cost tracking parameters
        self.prompt_token_cost: float = self.config.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = self.config.get("response_token_cost", 0.0)
        # Randomness configuration
        self.temperature: float = self.config.get("temperature", 1.0)
        self.max_tokens: int = self.config.get("max_tokens", 1024)
        self.stop: Union[str, List[str]] = self.config.get("stop", None)

        self.api_key: str = os.getenv(
            "OPENROUTER_API_KEY", self.config.get("api_key", "")
        )
        if self.api_key == "":
            raise ValueError(
                "OPENROUTER_API_KEY is not set in environment or configuration"
            )

        # OpenRouter-specific headers to identify the application
        headers = {
            "HTTP-Referer": "https://github.com/angrysky56/auto-graph-of-thoughts",
            "X-Title": "Auto Graph of Thoughts Framework",
        }

        # Initialize the OpenAI Client pointing to OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
        )
        self.aclient = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
        )

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the OpenRouter model for responses.
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    if next_try <= 0:
                        raise ValueError("next_try must be positive")
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        "Error in OpenRouter: %s, trying again with %d samples",
                        e,
                        next_try,
                    )
                    time.sleep(random.randint(1, 3))  # nosec B311
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    async def aquery(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Asynchronously query the OpenRouter model for responses.
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = await self.achat(
                [{"role": "user", "content": query}], num_responses
            )
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    if next_try <= 0:
                        raise ValueError("next_try must be positive")
                    res = await self.achat(
                        [{"role": "user", "content": query}], next_try
                    )
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        "Error in OpenRouter async: %s, trying again with %d samples",
                        e,
                        next_try,
                    )
                    await asyncio.sleep(random.randint(1, 3))  # nosec B311
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to OpenRouter and retrieve the model's response.
        Implements backoff on OpenAI error.
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            "Response from OpenRouter: %s\nCost of response: %s",
            response,
            self.cost,
        )
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def achat(
        self, messages: List[Dict], num_responses: int = 1
    ) -> ChatCompletion:
        """
        Send chat messages to OpenRouter asynchronously and retrieve the model's response.
        """
        response = await self.aclient.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            "Async response from OpenRouter: %s\nCost of response: %s",
            response,
            self.cost,
        )
        return response

    def get_response_texts(
        self, query_responses: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract response text from OpenRouter completion response.
        """
        if not isinstance(query_responses, List):
            query_responses = [query_responses]
        return [
            choice.message.content
            for response in query_responses
            for choice in response.choices
        ]
