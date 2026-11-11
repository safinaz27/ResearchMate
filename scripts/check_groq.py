"""Check that Groq is reachable and the configured models work.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\check_groq.py

Checks, in order:
  1. GROQ_API_KEY is configured (reports only yes/no, never the value).
  2. The live /models endpoint accepts the key and lists GROQ_MODEL and/or
     GROQ_FALLBACK_MODEL (model availability is checked live, not assumed).
  3. One tiny chat completion succeeds for each configured model that is listed.

This makes real (very small) API calls and needs the key. The pytest suite
never runs it.
"""

from __future__ import annotations

import sys
import time

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

from researchmate.config import get_settings
from researchmate.llm.groq_client import (
    GroqNotConfiguredError,
    create_groq_client,
    select_model,
)
from researchmate.logging_config import redact, register_secret

PROMPT = "Reply with exactly the single word: OK"


def say(text: str) -> None:
    """Print through the redactor, so an error message can never leak the key."""
    print(redact(text))


def check_models(client: OpenAI, preferred: str, fallback: str) -> list[str] | None:
    """Return the configured models that Groq lists, or None if the call failed."""
    start = time.perf_counter()
    ids = sorted(m.id for m in client.models.list())
    say(f"  /models returned {len(ids)} models in {time.perf_counter() - start:.2f}s")
    gpt_oss = [i for i in ids if "gpt-oss" in i]
    say(f"  gpt-oss models listed: {gpt_oss or 'none'}")
    for name, model in (("GROQ_MODEL", preferred), ("GROQ_FALLBACK_MODEL", fallback)):
        say(f"  {name:<20} {model:<24} {'available' if model in ids else 'NOT LISTED'}")
    chosen = select_model(ids, preferred, fallback)
    say(f"  model that will be used: {chosen}")
    if chosen is None:
        return None
    return [m for m in (preferred, fallback) if m in ids]


def check_completion(client: OpenAI, model: str) -> bool:
    """Send one tiny prompt and report latency, token usage and the reply."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        # gpt-oss models reason before answering; reasoning tokens count toward
        # this limit, so it must leave room for the visible answer too.
        max_completion_tokens=256,
        reasoning_effort="low",
        temperature=0,
    )
    elapsed = time.perf_counter() - start
    choice = response.choices[0]
    reply = (choice.message.content or "").strip()
    usage = response.usage
    tokens = (f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
              if usage else "usage=n/a")
    say(f"  {model}: {elapsed:.2f}s  finish_reason={choice.finish_reason}  {tokens}")
    say(f"  reply: {reply!r}")
    return bool(reply)


def main() -> int:
    settings = get_settings()
    say("[config]")
    say(f"  GROQ_API_KEY configured: {settings.groq_configured}")
    say(f"  base_url: {settings.groq_base_url}")
    try:
        client = create_groq_client(settings, max_retries=0)  # fail fast in a diagnostic
    except GroqNotConfiguredError as exc:
        say(f"  FAIL  {exc}")
        return 1
    register_secret(settings.groq_api_key.get_secret_value())  # type: ignore[union-attr]

    results: dict[str, bool] = {"config": True}
    try:
        say("[models]")
        usable = check_models(client, settings.groq_model, settings.groq_fallback_model)
        results["models"] = usable is not None
        for model in usable or []:
            say(f"[completion {model}]")
            results[f"completion:{model}"] = check_completion(client, model)
    except AuthenticationError as exc:
        say(f"  FAIL  key rejected by Groq (HTTP {exc.status_code}). Check GROQ_API_KEY.")
        results["auth"] = False
    except APIConnectionError as exc:
        say(f"  FAIL  cannot reach {settings.groq_base_url}: {exc}")
        results["network"] = False
    except APIStatusError as exc:
        say(f"  FAIL  HTTP {exc.status_code}: {exc.message}")
        results["api"] = False

    say("\nSUMMARY: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
