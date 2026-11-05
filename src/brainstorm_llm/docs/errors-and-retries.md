# Errors and retries

## Exception tree

```
BrainStormError
├── ConfigurationError
│   └── MissingAPIKeyError
├── APIError
│   ├── APIConnectionError
│   │   └── APITimeoutError
│   ├── APIStatusError
│   │   ├── AuthenticationError        (401)
│   │   ├── PermissionDeniedError      (403)
│   │   ├── BadRequestError            (400)
│   │   │   └── UnsupportedModelError
│   │   ├── RequestTooLargeError       (413)
│   │   ├── ZDRUnavailableError        (422 cmd_zdr_no_providers)
│   │   ├── RateLimitError             (429)
│   │   └── ServerError                (5xx / 529)
│   └── APIStreamError
├── ResponseValidationError
├── PricingUnavailableError
└── PricingParseError
```

`APIStatusError` exposes `status_code`, `error_type`, `code`, `param`,
`request_id`, `body`, and `retry_after`.

```python
from brainstorm_llm import RateLimitError, ZDRUnavailableError

try:
    llm.generate([{"role": "user", "content": "hi"}])
except RateLimitError as exc:
    print(exc.retry_after)
except ZDRUnavailableError:
    print("this model has no zero-data-retention upstream")
```

Errors are parsed from either envelope:

```json
{"error": {"message": "...", "type": "...", "code": "...", "param": "..."}}
```

```json
{"type": "error", "error": {"type": "...", "message": "..."}}
```

## Retries

Default policy:

- Maximum 2 retries.
- Retry statuses: `408, 409, 429, 500, 502, 503, 504, 529`.
- Connection and timeout errors are retried.
- `400, 401, 403, 404, 413, 422` are never retried.
- Exponential backoff with jitter, capped at 8 seconds.
- `Retry-After` is honored.

```python
llm = BrainStormLLM(max_retries=0)
```

Streaming requests are retried only before the first event is yielded. A failure
after streaming starts raises `APIStreamError` or the upstream status error.

## Timeouts

```python
llm = BrainStormLLM(timeout=120.0)
```

`timeout` applies to connect/read/write/pool via `httpx.Timeout`.

## ZDR

Command Code:

- `BrainStormLLM`: `x-cmd-zdr: 1` by default.
- `BrainStormS1`: off by default; `/systemone` has no ZDR-capable upstream and
  returns 422 when the header is sent.

Other providers ignore the ZDR setting.
