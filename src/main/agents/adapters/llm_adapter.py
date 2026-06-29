"""LLM Adapter ΓÇö wraps Azure OpenAI (GPT-4.1) for agent use.

Provides a simple `.invoke(system_prompt, user_prompt) -> str` interface
that all LLM-powered agents (Sentiment, Event, etc.) call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

from openai import OpenAI  # type: ignore


class LLMAdapter:
    """Thin wrapper around Azure OpenAI chat completions.

    Usage::

        llm = LLMAdapter(
            endpoint="https://ΓÇª.openai.azure.com/openai/v1",
            api_key="ΓÇª",
            model="gpt-4.1",
        )
        response_text = llm.invoke(
            system_prompt="You are a financial analyst.",
            user_prompt="Summarise TCS news.",
        )
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "gpt-4.1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """Send a chat completion request and return the assistant message text.

        Parameters
        ----------
        system_prompt : str
            System-level instruction.
        user_prompt : str
            User content (e.g. news articles to analyse).
        temperature : float, optional
            Override default temperature for this call.
        max_tokens : int, optional
            Override default max_tokens for this call.
        response_format : dict, optional
            E.g. ``{"type": "json_object"}`` to force JSON output.

        Returns
        -------
        str  ΓÇö raw text from the assistant message.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        completion = self._client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience: invoke and parse the response as JSON."""
        raw = self.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
            **kwargs,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw, "_parse_error": True}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any], base_path: Optional[str] = None) -> "LLMAdapter":
        """Build from a config dict (typically loaded from config.yaml).
        
        If config["llm"]["credentials_file"] is specified, loads endpoint/api_key
        from that file (which should be git-ignored for security).
        """
        import yaml
        
        llm_cfg = config.get("llm", {})
        
        # Load credentials from separate file if specified
        endpoint = llm_cfg.get("endpoint", "")
        api_key = llm_cfg.get("api_key", "")
        
        credentials_file = llm_cfg.get("credentials_file")
        if credentials_file:
            if base_path:
                creds_path = os.path.join(base_path, credentials_file)
            else:
                creds_path = credentials_file
            
            if os.path.exists(creds_path):
                with open(creds_path, "r", encoding="utf-8") as f:
                    creds = yaml.safe_load(f) or {}
                    creds_llm = creds.get("llm", {})
                    endpoint = creds_llm.get("endpoint", endpoint)
                    api_key = creds_llm.get("api_key", api_key)

        # Environment variables take precedence (used in cloud / containers).
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", endpoint)
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", api_key)

        return cls(
            endpoint=endpoint,
            api_key=api_key,
            model=llm_cfg.get("model", "gpt-4.1"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
