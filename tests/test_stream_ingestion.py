from database_storage import ingestion


class FakeStore:
    def __init__(self):
        self.records = []
        self.closed = False

    def upsert(self, record):
        self.records.append(record)

    def close(self):
        self.closed = True


class FakeGraph:
    def __init__(self):
        self.records = []
        self.closed = False

    def insert_observation(self, record):
        self.records.append(record)

    def close(self):
        self.closed = True


def test_enriched_stream_is_sent_to_sql_and_graph(monkeypatch):
    records = [{"id": "obs-1"}, {"id": "obs-2"}]
    monkeypatch.setattr(ingestion, "iter_jsonl", lambda path: iter(records))
    sql = FakeStore()
    graph = FakeGraph()

    count = ingestion.ingest_jsonl("unused.jsonl", sql_store=sql, graph=graph)

    assert count == 2
    assert sql.records == records
    assert graph.records == records
    # Objects supplied by callers can be reused for another batch.
    assert not sql.closed
    assert not graph.closed


def test_graph_reasoner_counts_spending_and_stops_at_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from processing.enrichment.backends import BudgetExceeded, UsageLedger
    from database_storage import reasoning

    monkeypatch.setattr(reasoning, "get_config", lambda: {"openai": {"api": "x"}})
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                               usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reasoner = reasoning.GraphReasoner(client=client,
                                       ledger=UsageLedger(tmp_path / "u.jsonl", budget_usd=0.1))
    assert reasoner.ask("first") == "{}"
    with pytest.raises(BudgetExceeded):
        reasoner.ask("second")
    assert len(calls) == 1
