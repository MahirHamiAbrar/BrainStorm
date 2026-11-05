from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "brainstorm_llm"
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(anthropic|openai|typesafe_sdk)\b", re.MULTILINE
)


def test_no_forbidden_provider_sdks_imported() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if FORBIDDEN_IMPORT.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == []


def test_public_api_imports() -> None:
    import brainstorm_llm as package

    for name in [
        "BrainStormLLM",
        "BrainStormS1",
        "Message",
        "MessageParam",
        "TextBlock",
        "ToolUseBlock",
        "ToolResultBlock",
        "Usage",
        "Noul",
        "Choice",
        "Score",
        "SystemOneResponse",
        "ModelInfo",
        "ModelPricing",
        "TokenPrice",
        "list_models",
        "model_prices",
        "CallUsage",
        "CostTracker",
    ]:
        assert hasattr(package, name), name
