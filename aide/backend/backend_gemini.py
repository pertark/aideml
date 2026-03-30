"""Backend for Google Gemini using the google-genai SDK.

Authentication:
- Set GOOGLE_GENAI_USE_VERTEXAI=True to use Vertex AI (picks up
  GOOGLE_APPLICATION_CREDENTIALS automatically via google-auth ADC).
- Otherwise uses GEMINI_API_KEY for the Gemini Developer API.
"""

import logging
import time

import backoff
from funcy import once
from google import genai
from google.genai import types

from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("aide")

_client: genai.Client = None  # type: ignore

try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        InternalServerError as _GoogInternalServerError,
        ResourceExhausted,
        ServiceUnavailable,
    )

    _RETRY_EXCEPTIONS = (
        ResourceExhausted,
        ServiceUnavailable,
        DeadlineExceeded,
        _GoogInternalServerError,
    )
except ImportError:
    _RETRY_EXCEPTIONS = (Exception,)


@once
def _setup_gemini_client():
    global _client
    _client = genai.Client()


def _to_str(msg: PromptType) -> str:
    """Ensure a prompt (str, dict, or list) is a plain string."""
    if isinstance(msg, str):
        return msg
    return compile_prompt_to_md(msg)


def _func_spec_to_tool(func_spec: FunctionSpec) -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=func_spec.name,
                description=func_spec.description,
                parameters=func_spec.json_schema,
            )
        ]
    )


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_gemini_client()

    model = model_kwargs.pop("model")
    temperature = model_kwargs.pop("temperature", None)
    max_tokens = model_kwargs.pop("max_tokens", None)

    # Gemini requires non-empty user content; if only a system message was
    # provided, promote it to the user turn (matches prior behaviour).
    if system_message is not None and user_message is None:
        contents = _to_str(system_message)
        system_instruction = None
    else:
        contents = _to_str(user_message) if user_message is not None else ""
        system_instruction = _to_str(system_message) if system_message is not None else None

    config_kwargs: dict = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    if func_spec is not None:
        config_kwargs["tools"] = [_func_spec_to_tool(func_spec)]
        config_kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[func_spec.name],
            )
        )

    config = types.GenerateContentConfig(**config_kwargs)

    @backoff.on_exception(backoff.expo, _RETRY_EXCEPTIONS, max_value=60, factor=1.5)
    def _call():
        return _client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    logger.info(f"Gemini API request: model={model}, system={system_message}, user={user_message}")

    t0 = time.time()
    response = _call()
    req_time = time.time() - t0

    candidate = response.candidates[0]
    parts = candidate.content.parts if candidate.content else []
    first_part = parts[0] if parts else None

    if func_spec is not None and first_part is not None and first_part.function_call:
        fc = first_part.function_call
        if fc.name != func_spec.name:
            logger.warning(
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {fc.name}. Falling back to text."
            )
            output = response.text
        else:
            output = dict(fc.args)
    else:
        if func_spec is not None:
            logger.warning(
                "No function call in response despite func_spec. "
                f"Falling back to text.\nContent: {response.text}"
            )
        output = response.text

    in_tokens = response.usage_metadata.prompt_token_count or 0
    out_tokens = response.usage_metadata.candidates_token_count or 0

    info = {
        "model": model,
        "finish_reason": str(candidate.finish_reason),
    }

    logger.info(
        f"Gemini API call completed - {model} - {req_time:.2f}s - "
        f"{in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"Gemini API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
