from __future__ import annotations

from brainstorm_llm import ModelInfo, list_models
from brainstorm_llm.models import PUBLIC_CATALOG_PROVIDERS


class FakeProvider:
    def __init__(self, models):
        self.models = models
        self.closed = False
        self.kwargs = {}

    def list_models(self, **kwargs):
        return self.models

    def close(self):
        self.closed = True


def test_module_level_list_models(monkeypatch) -> None:
    fake = FakeProvider([ModelInfo(id="m", name="M", provider="commandcode")])
    captured = {}

    def fake_get_provider(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr("brainstorm_llm.models.get_provider", fake_get_provider)
    models = list_models("commandcode", include_decision_models=False)
    assert models[0].id == "m"
    assert fake.closed is True
    assert captured["kwargs"]["api_key"] == "catalog-only"


def test_model_info_has_no_pricing_fields() -> None:
    fields = set(ModelInfo.model_fields)
    assert not any("price" in field or "cost" in field for field in fields)


def test_public_catalog_providers() -> None:
    assert {"commandcode", "openrouter", "opencode"} <= PUBLIC_CATALOG_PROVIDERS
