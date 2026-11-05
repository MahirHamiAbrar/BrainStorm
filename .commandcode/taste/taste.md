# Taste

- Prefers Anthropic Messages-style schemas as the project's canonical internal shape, with OpenAI-style provider responses converted into that shape rather than exposing per-provider types. Confidence: 0.85
- Avoids vendor SDK dependencies (anthropic, openai, typesafe-sdk) and prefers hand-rolled HTTP via httpx, with both sync and async APIs supported. Confidence: 0.85
- Expects every public function, class, and schema a user touches first (clients, methods, dataclasses/models, exceptions, helpers) to carry a docstring documenting purpose, key args, returns, and raises — not just types. Confidence: 0.7
- For framework integrations, wants the assistant to research the current upstream API/docs first and mirror the official interface (and existing official adapters) instead of inventing an incompatible one. Confidence: 0.7
- New functionality must ship with tests. Confidence: 0.65
