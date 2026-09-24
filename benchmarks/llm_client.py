"""Cached LLM calls for the benchmark runners.

Every response is stored under the cache directory (benchmarks/.cache/ by default, git-ignored),
keyed by provider, model, generation settings and the full prompt, so re-running a benchmark or
resuming it after a quota error never repeats a call that already succeeded. Live calls are also
appended to live_calls.jsonl in the cache directory, which is how the daily call cap is enforced.

Providers
    gemini   Google AI Studio through google-generativeai (the SDK the rest of the repo uses);
             key from GEMINI_API_KEY or GOOGLE_API_KEY.
    ollama   a local Ollama server's native /api/generate endpoint (so the context window can be
             set per request); base URL from OLLAMA_HOST, default http://127.0.0.1:11434.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache"


class LLMStop(RuntimeError):
    """A reason to stop making LLM calls in this run (the runners save partial results)."""


class QuotaExhausted(LLMStop):
    """The provider refused with a quota / rate-limit error, or a local cap was reached."""


class LLMCallFailed(LLMStop):
    """Any other provider error (unknown model, bad request, network failure)."""


class LLMClient:
    def __init__(self, provider: str, model: str, cache_dir=DEFAULT_CACHE, min_interval_s: float = 0.0,
                 max_live_calls: int = None, daily_cap: int = None, num_ctx: int = 16384,
                 temperature: float = 0.0, offline: bool = False, timeout_s: float = 600.0):
        if provider not in ("gemini", "ollama"):
            raise ValueError(f"unknown provider {provider!r}")
        self.provider, self.model = provider, model
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.max_live_calls, self.daily_cap = max_live_calls, daily_cap
        self.num_ctx, self.temperature = num_ctx, temperature
        self.offline = offline          # cache only: a miss raises instead of calling the model
        self.timeout_s = timeout_s      # client deadline per call (600 s is the SDK default)
        self.live_calls = 0
        self._last_call = 0.0
        self._gemini = None

    # ------------------------------------------------------------------ cache
    def _key(self, prompt: str, json_mode: bool) -> str:
        blob = json.dumps({"provider": self.provider, "model": self.model, "json": json_mode,
                           "temperature": self.temperature,
                           "num_ctx": self.num_ctx if self.provider == "ollama" else None,
                           "prompt": prompt}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / self.provider / self.model.replace("/", "_") / f"{key}.json"

    def cached(self, prompt: str, json_mode: bool = True):
        p = self._path(self._key(prompt, json_mode))
        return json.loads(p.read_text()) if p.exists() else None

    def _ledger(self) -> Path:
        return self.cache_dir / "live_calls.jsonl"

    def calls_today(self) -> int:
        if not self._ledger().exists():
            return 0
        today = datetime.now().date().isoformat()
        n = 0
        for line in self._ledger().read_text().splitlines():
            rec = json.loads(line)
            if rec["provider"] == self.provider and rec["time"].startswith(today):
                n += 1
        return n

    # ------------------------------------------------------------------ calls
    def complete(self, prompt: str, json_mode: bool = True) -> dict:
        """Return {"text", "prompt_tokens", "completion_tokens", "latency_s", "cached"}."""
        key = self._key(prompt, json_mode)
        path = self._path(key)
        if path.exists():
            rec = json.loads(path.read_text())
            rec["cached"] = True
            return rec
        if self.offline:
            raise QuotaExhausted("offline mode: response not in cache")
        if self.max_live_calls is not None and self.live_calls >= self.max_live_calls:
            raise QuotaExhausted(f"--max-live-calls {self.max_live_calls} reached")
        if self.daily_cap is not None and self.calls_today() >= self.daily_cap:
            raise QuotaExhausted(f"daily cap of {self.daily_cap} {self.provider} calls reached")
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()
        t0 = time.time()
        status = "ok"
        try:
            if self.provider == "gemini":
                rec = self._call_gemini(prompt, json_mode)
            else:
                rec = self._call_ollama(prompt, json_mode)
        except QuotaExhausted:
            status = "quota"
            raise
        except Exception as exc:
            status = f"error: {type(exc).__name__}"
            raise LLMCallFailed(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        finally:
            self.live_calls += 1
            self._ledger().parent.mkdir(parents=True, exist_ok=True)
            with self._ledger().open("a") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                    "provider": self.provider, "model": self.model,
                                    "prompt_chars": len(prompt), "status": status}) + "\n")
        rec["latency_s"] = round(time.time() - t0, 3)
        rec.update({"provider": self.provider, "model": self.model,
                    "time": datetime.now().isoformat(timespec="seconds")})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        rec["cached"] = False
        return rec

    def _call_gemini(self, prompt, json_mode):
        import google.generativeai as genai

        if self._gemini is None:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
            genai.configure(api_key=key)
            self._gemini = genai.GenerativeModel(self.model)
        kwargs = {"response_mime_type": "application/json"} if json_mode else {}
        if self.temperature is not None:        # None keeps the model's default, as prototype1/agent.py does
            kwargs["temperature"] = self.temperature
        cfg = genai.types.GenerationConfig(**kwargs)
        try:
            resp = self._gemini.generate_content(prompt, generation_config=cfg,
                                                 request_options={"timeout": self.timeout_s})
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "ResourceExhausted" in type(exc).__name__:
                raise QuotaExhausted(" ".join(msg.split())[:800]) from exc
            raise
        usage = getattr(resp, "usage_metadata", None)
        return {"text": resp.text,
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "thinking_tokens": getattr(usage, "thoughts_token_count", None) or 0}

    def _call_ollama(self, prompt, json_mode):
        import requests

        base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not base.startswith("http"):
            base = "http://" + base
        options = {"num_ctx": self.num_ctx, "seed": 0}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        if json_mode:
            body["format"] = "json"
        for attempt in range(3):   # a local runner can crash (e.g. memory pressure) and restart
            try:
                r = requests.post(f"{base}/api/generate", json=body, timeout=max(self.timeout_s, 1800))
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.HTTPError):
                if attempt == 2:
                    raise
                time.sleep(30)
        data = r.json()
        prompt_tokens = data.get("prompt_eval_count")
        return {"text": data.get("response", ""), "prompt_tokens": prompt_tokens,
                "completion_tokens": data.get("eval_count"),
                "context_limit_hit": bool(prompt_tokens and prompt_tokens >= self.num_ctx - 1)}


def parse_json(text):
    """Parse a JSON reply, tolerating Markdown code fences. Returns None if it is not JSON."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None
