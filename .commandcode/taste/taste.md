# Taste

- Prefers Anthropic Messages-style schemas as the project's canonical internal shape, with OpenAI-style provider responses converted into that shape rather than exposing per-provider types. Confidence: 0.85
- Avoids vendor SDK dependencies (anthropic, openai, typesafe-sdk) and prefers hand-rolled HTTP via httpx, with both sync and async APIs supported. Confidence: 0.85
