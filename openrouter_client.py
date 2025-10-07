
import os
import json
from typing import List, Dict, Optional, Generator, Union
import requests
from keys import OPENROUTER_KEY

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterError(Exception):
    pass

def _build_headers(referer: Optional[str], title: Optional[str], api_key: Optional[str]) -> Dict[str, str]:
    key = OPENROUTER_KEY
    if not key:
        raise OpenRouterError("Missing OPENROUTER_API_KEY. Set env var or pass api_key.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers

def chat_complete(
    model: str,
    messages: List[Dict[str, str]],
    prompt: str = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 1024,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> str:
    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    if not messages:
        raise OpenRouterError("Need either messages or prompt.")
    """
    Non-streaming convenience wrapper. Returns assistant text.
    messages: list of {role: "system"|"user"|"assistant", content: "..."}
    """
    headers = _build_headers(referer, title, api_key)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.RequestException as e:
        raise OpenRouterError(f"Request failed: {e}") from e

    if resp.status_code != 200:
        raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception:
        raise OpenRouterError(f"Non-JSON response: {resp.text[:500]}")

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError(f"Unexpected response schema: {json.dumps(data)[:1200]}") from e

def chat_complete_raw(
    model: str,
    messages: List[Dict[str, str]],
    **kwargs,
) -> Dict:
    """
    Same as chat_complete but returns raw JSON dict from OpenRouter.
    """
    headers = _build_headers(kwargs.get("referer"), kwargs.get("title"), kwargs.get("api_key"))
    payload = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 1.0),
        "max_tokens": kwargs.get("max_tokens", 1024),
        "stream": False,
    }
    timeout = kwargs.get("timeout", 120)
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.RequestException as e:
        raise OpenRouterError(f"Request failed: {e}") from e

    if resp.status_code != 200:
        raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()

def stream_chat_complete(
    model: str,
    messages: List[Dict[str, str]],
    prompt: str = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 300,
) -> Generator[str, None, None]:
    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    if not messages:
        raise OpenRouterError("Need either messages or prompt.")
    """
    Streaming generator. Yields text chunks as they arrive.
    Note: OpenRouter uses SSE with JSON lines prefixed by 'data: '.
    """
    headers = _build_headers(referer, title, api_key)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        with requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                raise OpenRouterError(f"HTTP {r.status_code}: {r.text}")
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    # Ignore malformed lines
                    continue
    except requests.RequestException as e:
        raise OpenRouterError(f"Stream request failed: {e}") from e
