"""Module for interfacing with OpenAI LLMs."""

import logging
from typing import Optional

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import config

logger = logging.getLogger(__name__)

def create_openai_embedding() -> OpenAIEmbedding:
    """Creates an OpenAI Embedding model for vector representation.

    Returns:
        OpenAIEmbedding model.
    """
    openai_embedding = OpenAIEmbedding(model=config.EMBEDDING_MODEL_ID)
    logger.info(f"Created OpenAI Embedding model: {config.EMBEDDING_MODEL_ID}")
    return openai_embedding

def create_openai_llm(
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None
) -> OpenAI:
    """Creates an OpenAI LLM for generating responses.

    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        OpenAI LLM model.
    """
    openai_llm = OpenAI(
        model=config.LLM_MODEL_ID,
        temperature=temperature if temperature is not None else config.TEMPERATURE,
        max_tokens=max_new_tokens if max_new_tokens is not None else config.MAX_NEW_TOKENS,
        additional_kwargs={"top_p": config.TOP_P},
    )
    logger.info(f"Created OpenAI LLM model: {config.LLM_MODEL_ID}")
    return openai_llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.

    Args:
        new_model_id: New LLM model ID to use.
    """
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"LLM model changed to: {new_model_id}")
