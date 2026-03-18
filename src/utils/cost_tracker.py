"""LLM Cost Tracker — automatic token and cost accumulation via LangChain callbacks.

Usage:
    from src.utils.cost_tracker import cost_tracker

    llm = ChatOpenAI(model="gpt-5.4-nano", callbacks=[cost_tracker])
    llm.invoke(prompt)  # cost_tracker.on_llm_end fires automatically

    cost_tracker.print_summary()
"""

import threading
from collections import defaultdict
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from config.settings import MODEL_PRICING, USD_TO_INR


class CostTracker(BaseCallbackHandler):
    """Accumulates token usage and cost across all LLM calls.

    Thread-safe — safe for use with asyncio.gather and ThreadPoolExecutor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called after every LLM call. Extracts token usage from response."""
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        model = llm_output.get("model_name", "unknown")

        with self._lock:
            entry = self._usage[model]
            entry["calls"] += 1
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

    def _resolve_pricing(self, model: str) -> dict | None:
        """Return pricing entry for *model*, tolerating date-suffixed names.

        OpenAI sometimes returns names like ``gpt-5.4-nano-2026-03-17`` while
        MODEL_PRICING keys are ``gpt-5.4-nano``.  We first try an exact match,
        then fall back to the longest prefix match.
        """
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]
        # Longest-prefix fallback (e.g. "gpt-5.4-nano" matches "gpt-5.4-nano-2026-03-17")
        best_key = max(
            (k for k in MODEL_PRICING if model.startswith(k)),
            key=len,
            default=None,
        )
        return MODEL_PRICING.get(best_key) if best_key else None

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost for a model's token usage."""
        pricing = self._resolve_pricing(model)
        if not pricing:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured cost summary."""
        with self._lock:
            per_model = {}
            total_usd = 0.0
            total_input = 0
            total_output = 0
            total_calls = 0

            for model, usage in sorted(self._usage.items()):
                cost_usd = self._compute_cost(model, usage["input_tokens"], usage["output_tokens"])
                total_usd += cost_usd
                total_input += usage["input_tokens"]
                total_output += usage["output_tokens"]
                total_calls += usage["calls"]
                per_model[model] = {
                    "calls": usage["calls"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cost_usd": round(cost_usd, 8),
                    "cost_inr": round(cost_usd * USD_TO_INR, 6),
                }

            return {
                "per_model": per_model,
                "total_calls": total_calls,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cost_usd": round(total_usd, 8),
                "total_cost_inr": round(total_usd * USD_TO_INR, 6),
            }

    def print_summary(self) -> None:
        """Print a formatted cost summary table to console."""
        summary = self.get_summary()

        if not summary["per_model"]:
            print("\nNo LLM calls recorded.")
            return

        print("\n" + "=" * 80)
        print("PIPELINE COST SUMMARY")
        print("=" * 80)
        header = f"{'Model':<20} {'Calls':>6} {'Input Tok':>12} {'Output Tok':>12} {'Cost (USD)':>15} {'Cost (INR)':>14}"
        print(header)
        print("-" * 80)

        for model, data in summary["per_model"].items():
            print(
                f"{model:<20} {data['calls']:>6} "
                f"{data['input_tokens']:>12,} {data['output_tokens']:>12,} "
                f"${data['cost_usd']:>14.8f} "
                f"Rs {data['cost_inr']:>13.6f}"
            )

        print("-" * 80)
        s = summary
        print(
            f"{'TOTAL':<20} {s['total_calls']:>6} "
            f"{s['total_input_tokens']:>12,} {s['total_output_tokens']:>12,} "
            f"${s['total_cost_usd']:>14.8f} "
            f"Rs {s['total_cost_inr']:>13.6f}"
        )
        print("=" * 80)

    def track_tts(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        """Manually record a Vertex AI / ElevenLabs TTS API call.

        Call this after each successful TTS synthesis since TTS providers
        don't use LangChain callbacks.
        """
        if input_tokens == 0 and output_tokens == 0:
            return
        with self._lock:
            entry = self._usage[model]
            entry["calls"] += 1
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

    def reset(self) -> None:
        """Clear all accumulated data. Useful for test isolation."""
        with self._lock:
            self._usage.clear()


# Module-level singleton — import this in agent files
cost_tracker = CostTracker()
