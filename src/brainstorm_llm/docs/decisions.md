# Decision models (`BrainStormS1`)

`BrainStormS1` calls the TypeSafe **System One** API. It returns typed
probabilities, never text, and never streams.

```python
from brainstorm_llm import BrainStormS1, Noul, Choice, Score

s1 = BrainStormS1(
    provider="commandcode",
    model="typesafe/jev-1.14.0",   # default
)
```

## Question types

```python
questions = {
    "urgent": Noul(
        instructions="Does this need urgent attention?",
        criteria={"true": "Same-day resolution required", "false": "Can wait"},
    ),
    "team": Choice(
        instructions="Which team should handle it?",
        criteria={
            "billing": "charges and payments",
            "shipping": "delivery and tracking",
            "returns": "refunds and exchanges",
        },
    ),
    "frustration": Score(
        instructions="How frustrated is the customer?",
        criteria=["Calm", "Frustrated", "Very angry"],
    ),
}
```

- `Noul` → probability of yes (`0..1`).
- `Choice` → selected label, confidence, and per-option probabilities.
- `Score` → expected score, confidence, rubric `legend`, and per-level probabilities.

A request takes up to 100 questions sharing one `state`. `state` may be a string,
dict, or list.

## Response

```python
result = s1.system_one(state, questions)

result.nouls["urgent"].noul                  # 0.96
result.choices["team"].choice                # "billing"
result.choices["team"].probabilities         # {"billing": 0.91, ...}
result.scores["frustration"].score           # 1.41
result.usage                                 # input_tokens / output_tokens
result.request_id                            # x-typesafe-request-id when present
```

## Typed response model

Subclass `SystemOneResponse` and each answer becomes a field:

```python
from brainstorm_llm import ChoiceAnswer, NoulAnswer, SystemOneResponse

class BillingResponse(SystemOneResponse):
    urgent: NoulAnswer
    team: ChoiceAnswer

result = s1.system_one(
    state,
    {"urgent": Noul(instructions="urgent?"), ...},
    response_model=BillingResponse,
)
assert 0 <= result.urgent.noul <= 1
```

A plain Pydantic `BaseModel` can also be passed; it validates the raw response
(`model`, `answers`, `usage`).

## Images

```python
from brainstorm_llm import Image, Noul

result = s1.system_one(
    state="The customer returned this item: <photo>",
    questions={"damaged": Noul(instructions="Is the item damaged?")},
    assets={"<photo>": Image("return.jpg")},
)
```

Assets are sent as base64 JSON. Image questions require a vision-capable model.

## Async

```python
async with BrainStormS1() as s1:
    result = await s1.asystem_one(state, questions)
```

## Available models

`BrainStormS1` also exposes a TypeSafe-SDK-style model resource:

```python
for model in s1.models.list():
    print(model.id, model.kind)
```

## Providers

| Provider | Base URL | Notes |
|---|---|---|
| `commandcode` | `https://api.commandcode.ai/provider/v1` | Default; ZDR off |
| `typesafe` | `https://api.typesafe.ai/v1` | Direct TypeSafe |
| `opencode` | `https://opencode.ai/inference` | `jev-1.13` models |

## Limits

Up to 100 questions per request. The model budget is 64K tokens for state plus all
questions, and 32K for state plus the longest single question.
