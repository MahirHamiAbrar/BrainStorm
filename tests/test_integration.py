from __future__ import annotations

import os

import pytest

from brainstorm_llm import BrainStormLLM, BrainStormS1, Choice, Noul, list_models, model_prices

pytestmark = pytest.mark.integration

COMMAND_KEY = os.getenv("COMMAND_API_KEY") or os.getenv("CMD_API_KEY")
requires_commandcode = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1" or not COMMAND_KEY,
    reason="set RUN_INTEGRATION=1 and COMMAND_API_KEY to run live tests",
)


@requires_commandcode
def test_commandcode_catalog() -> None:
    models = list_models("commandcode")
    assert any(model.id == "deepseek/deepseek-v4.1-flash" for model in models)
    assert any(model.kind == "decision" for model in models)


@requires_commandcode
def test_commandcode_generate_and_stream() -> None:
    llm = BrainStormLLM(model="deepseek/deepseek-v4.1-flash", pricing="off")
    try:
        message = llm.generate(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        )
        assert message.content
        assert message.usage.input_tokens > 0

        final = llm.stream_collect(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        )
        assert final.content
    finally:
        llm.close()


@requires_commandcode
def test_commandcode_decisions() -> None:
    s1 = BrainStormS1(pricing="off")
    try:
        result = s1.system_one(
            "My payments have failed for three days.",
            {
                "urgent": Noul(instructions="Is this urgent?"),
                "team": Choice(
                    instructions="Which team should handle it?",
                    criteria={"billing": "payments", "shipping": "delivery"},
                ),
            },
        )
        assert 0 <= result.nouls["urgent"].noul <= 1
        assert result.choices["team"].choice in {"billing", "shipping"}
    finally:
        s1.close()


@requires_commandcode
def test_commandcode_live_prices(tmp_path) -> None:
    pricing = model_prices("commandcode", "live", cache_dir=tmp_path)
    assert pricing.pricing_type == "live"
    assert "deepseek/deepseek-v4.1-flash" in pricing.prices


@requires_commandcode
def test_langchain_chat_and_decisions() -> None:
    from brainstorm_llm.integrations.langchain import (
        BrainStormDecisions,
        ChatBrainStorm,
        Predicate,
    )

    chat = ChatBrainStorm(model="deepseek/deepseek-v4.1-flash", pricing="off", max_tokens=256)
    try:
        reply = chat.invoke("Reply with exactly: LANGCHAIN_OK")
        assert reply.content_blocks
        assert reply.usage_metadata is not None
    finally:
        chat.close()

    decisions = BrainStormDecisions(pricing="off")
    try:
        result = decisions.invoke(
            {
                "input": "Production is down.",
                "questions": {"urgent": Predicate(instructions="Is it urgent?")},
            }
        )
        assert 0 <= result.predicates["urgent"].probability <= 1
    finally:
        decisions.close()
