"""OpenAI client used only for graph-context identity and incident decisions."""

from __future__ import annotations

from openai import OpenAI
from processing.enrichment.backends import UsageLedger

from .config import get_config


class GraphReasoner:
    def __init__(self, client=None, model: str | None = None, ledger: UsageLedger | None = None):
        settings = get_config()["openai"]
        self.client = client or OpenAI(api_key=settings["api"])
        self.model = model or settings.get("model", "gpt-4o-mini")
        # With openai.budget_usd set, calls stop once it is spent; callers
        # already treat a reasoning error as "skip this decision".
        if ledger is None and settings.get("budget_usd") is not None:
            ledger = UsageLedger(settings.get("usage_ledger", "/state/openai-usage.jsonl"),
                                 budget_usd=settings["budget_usd"])
        self.ledger = ledger

    def ask(self, prompt: str) -> str:
        if self.ledger is not None: self.ledger.check(self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if self.ledger is not None and response.usage is not None:
            self.ledger.record(self.model, "graph", response.usage)
        return response.choices[0].message.content or ""
