"""
OpenRouter Client for DSPy

A custom DSPy language model client that interfaces with OpenRouter's API.
Supports vision models, reasoning tokens, and provider routing.
"""

import dspy
import os
import sys
import requests
from ratelimit import limits

# Rate limiting configuration (430 calls per 10 seconds)
RL_CALLS = 430
RL_PERIOD_SECONDS = 10


class OpenRouterClient(dspy.LM):
    """DSPy-compatible client for OpenRouter chat completions.

    Supports:
      - Vision models with image inputs
      - Reasoning tokens (for models like o1/o3)
      - Provider routing and preferences
      - Standard OpenAI chat completion parameters
    """

    def __init__(
        self,
        api_key=None,
        model="openai/gpt-5-nano",
        base_url="https://openrouter.ai/api/v1",
        extra_headers=None,
        temperature=1.0,
        max_tokens=32000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout: int | float | None = None,
        reasoning_effort=None,
        reasoning: dict | None = None,
        reasoning_max_tokens: int | None = None,
        reasoning_enabled: bool | None = None,
        reasoning_exclude: bool | None = None,
        include_reasoning: bool | None = None,
        n=1,
        provider=None,  # <-- NEW: default provider routing for this client
        **kwargs,
    ):
        # Satisfy DSPy's internal assertion for "reasoning" models by passing 1.0 temp and >=20k tokens.
        super().__init__(
            model=model, api_key=None, temperature=1.0, max_tokens=max_tokens
        )

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.api_key = input("Enter your OpenRouter API key: ").strip()
        if not self.api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)

        self.model = model
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.reasoning_effort = reasoning_effort
        self.reasoning = reasoning
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_exclude = reasoning_exclude
        self.include_reasoning = include_reasoning

        # HTTP request timeout (seconds). Defaults to 120s; can be overridden via env.
        # Env var takes effect only if constructor argument is not provided.
        self.request_timeout = (
            float(os.getenv("OPENROUTER_REQUEST_TIMEOUT", "120"))
            if request_timeout is None
            else float(request_timeout)
        )

        # defaults used on each call (can be overridden by kwargs)
        self.req_defaults = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "provider": provider,  # <-- NEW: store default provider prefs
            "request_timeout": self.request_timeout,
            "reasoning": reasoning,
            "reasoning_effort": reasoning_effort,
            "reasoning_max_tokens": reasoning_max_tokens,
            "reasoning_enabled": reasoning_enabled,
            "reasoning_exclude": reasoning_exclude,
            "include_reasoning": include_reasoning,
        }

    def _headers(self):
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter telemetry / routing hints
        if "OPENROUTER_REFERER" in os.environ:
            h["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
        if "OPENROUTER_TITLE" in os.environ:
            h["X-Title"] = os.environ["OPENROUTER_TITLE"]
        h.update(self.extra_headers)
        return h

    @limits(calls=RL_CALLS, period=RL_PERIOD_SECONDS)
    def _request(self, messages, **overrides):
        """Make a rate-limited request to OpenRouter's chat completion endpoint."""
        # Merge defaults + overrides. If `provider` is given here, it will override client default.
        payload = {**self.req_defaults, **overrides}

        data = {
            "model": self.model,
            "messages": messages,
        }

        # Copy known OpenAI-style params through:
        for k in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "n",
        ):
            v = payload.get(k, None)
            if v is not None:
                data[k] = v

        # --- NEW: pass provider routing directly to OpenRouter ---
        if payload.get("provider") is not None:
            data["provider"] = payload["provider"]

        # You can also pass response_format, tool_choice, tools, etc. via overrides if needed
        for k, v in payload.items():
            if k in ("response_format", "tool_choice", "tools") and v is not None:
                data[k] = v

        # --- NEW: Unified reasoning parameter support (OpenRouter reasoning tokens) ---
        # Priority: explicit `reasoning` dict override wins. Otherwise, build from helpers.
        reasoning_cfg = payload.get("reasoning")
        if reasoning_cfg is None:
            # Build a reasoning object from individual helpers if any were specified
            r: dict[str, object] = {}
            eff = payload.get("reasoning_effort")
            if eff in ("high", "medium", "low"):
                r["effort"] = eff
            if payload.get("reasoning_max_tokens") is not None:
                r["max_tokens"] = int(payload["reasoning_max_tokens"])
            if payload.get("reasoning_enabled") is not None:
                r["enabled"] = bool(payload["reasoning_enabled"])
            # Legacy: include_reasoning => {} or { exclude: true }
            if payload.get("include_reasoning") is not None:
                inc = bool(payload["include_reasoning"])
                r["exclude"] = not inc
            if payload.get("reasoning_exclude") is not None:
                r["exclude"] = bool(payload["reasoning_exclude"])
            if r:
                reasoning_cfg = r
        if reasoning_cfg is not None:
            data["reasoning"] = reasoning_cfg

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=data,
            timeout=payload.get("request_timeout", self.request_timeout),
        )
        resp.raise_for_status()
        return resp.json()

    def __call__(self, messages=None, prompt=None, **kwargs):
        # Allow both messages=[...] or prompt="..."
        # write messages and prompt to file
        if messages is None:
            if prompt is None:
                raise TypeError(
                    "OpenRouterClient.__call__ requires 'messages' or 'prompt'."
                )
            messages = [{"role": "user", "content": str(prompt)}]

        payload = self._request(messages, **kwargs)
        choices = payload.get("choices", [])

        # Return a list[str] (DSPy-friendly)
        return [c["message"]["content"] for c in choices] if choices else [""]
