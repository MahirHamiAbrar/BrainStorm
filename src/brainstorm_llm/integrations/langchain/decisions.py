"""LangChain Runnable for System One decision models.

This mirrors LangChain's decisions integrations: a Runnable accepts typed
questions and returns probabilities, choices, and scores instead of a chat
message. It is also usable as a LangGraph node, or as a tool via
:meth:`BrainStormDecisions.as_tool`.
"""

from __future__ import annotations

import base64
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, TypedDict, cast

import orjson
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from ...decisions import BrainStormS1
from ...errors import BrainStormError
from ...schemas import (
    Choice as SystemOneChoice,
    ChoiceAnswer as SystemOneChoiceAnswer,
    Image,
    Noul as SystemOneNoul,
    NoulAnswer,
    Score as SystemOneScore,
    ScoreAnswer as SystemOneScoreAnswer,
)
from ._errors import raise_langchain_error

logger = logging.getLogger(__name__)

_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)
StateValue: TypeAlias = Any
State: TypeAlias = str | BaseMessage | Sequence[StateValue] | dict[str, StateValue]


class Predicate(BaseModel):
    """Estimate the probability that a condition is true."""

    type: Literal["predicate"] = "predicate"
    instructions: str = Field(min_length=1)


class Choice(BaseModel):
    """Select one value from a fixed, unordered set of options."""

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    choices: dict[str | bool, str | None] = Field(min_length=1)


class Level(BaseModel):
    """One level of a score rubric."""

    label: str = Field(min_length=1)
    description: str | None = None


class Score(BaseModel):
    """Rate the input against ordered levels."""

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    levels: list[str | Level] = Field(min_length=2)


Question = Annotated[Predicate | Choice | Score, Field(discriminator="type")]


class DecisionRequest(TypedDict):
    """Input mapping for one decision request.

    Use ``{"input": <state>, "questions": {"name": <question>}}``.
    """

    input: State
    questions: dict[str, Question]


class PredicateAnswer(BaseModel):
    """Probability that a predicate condition is true."""

    type: Literal["predicate"] = "predicate"
    probability: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    """Selected choice, its distribution, and confidence."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    """Expected rubric score with its legend and distribution."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[int, str] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class RefusalAnswer(BaseModel):
    """The model declined to answer a question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["refusal"] = "refusal"


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


class Usage(BaseModel):
    """Token usage reported for a decision request."""

    input_tokens: int | None = None
    output_tokens: int | None = None


class DecisionResponse(BaseModel):
    """Typed answers and metadata returned by a decision request."""

    model: str
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None

    @property
    def predicates(self) -> dict[str, PredicateAnswer]:
        """Predicate answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, PredicateAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Choice answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Score answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }

    @property
    def refusals(self) -> dict[str, RefusalAnswer]:
        """Refusal answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, RefusalAnswer)
        }


def _normalize_mime(mime_type: Any) -> str:
    mime = str(mime_type or "image/png").lower()
    if mime == "image/jpeg":
        return "jpg"
    if mime == "image/png":
        return "png"
    raise ValueError(
        f"System One image assets support PNG and JPEG; got {mime!r}. "
        "Convert the image before sending it."
    )


def _image_from_block(block: dict[str, Any]) -> Image | None:
    block_type = block.get("type")
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            return Image(base64.b64decode(str(source.get("data") or "")), format=_normalize_mime(source.get("media_type")))
        if block.get("base64"):
            return Image(
                base64.b64decode(str(block["base64"])),
                format=_normalize_mime(block.get("mime_type")),
            )
        if block.get("url"):
            raise ValueError("System One decision assets require base64 images, not hosted URLs.")
    if block_type in {"image_url", "input_image"}:
        raw = block.get("image_url")
        if isinstance(raw, dict):
            url = raw.get("url")
        elif isinstance(raw, str):
            url = raw
        else:
            url = None
        if isinstance(url, str):
            match = _DATA_URL.match(url)
            if match is None:
                raise ValueError(
                    "System One decision assets require base64 data URLs, not hosted URLs."
                )
            return Image(
                base64.b64decode(match.group(2)),
                format=_normalize_mime(match.group(1)),
            )
    return None


def _state_value(
    value: Any,
    assets: dict[str, Image],
    *,
    in_message: bool = False,
) -> Any:
    if isinstance(value, BaseMessage):
        content = _state_value(value.content, assets, in_message=True)
        if isinstance(value, HumanMessage):
            return content
        result: dict[str, Any] = {"role": value.type, "content": content}
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            result["tool_call_id"] = tool_call_id
        return result

    if isinstance(value, dict):
        image = _image_from_block(value)
        if image is not None:
            placeholder = f"<brainstorm-image-{uuid.uuid4().hex}>"
            assets[placeholder] = image
            return placeholder
        return {
            str(key): _state_value(item, assets, in_message=in_message)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_state_value(item, assets, in_message=in_message) for item in value]

    return value


def _to_state(value: Any) -> tuple[Any, dict[str, Image]]:
    assets: dict[str, Image] = {}
    return _state_value(value, assets), assets


def _to_systemone_question(question: Any) -> Any:
    if isinstance(question, (SystemOneNoul, SystemOneChoice, SystemOneScore)):
        return question
    if isinstance(question, Predicate):
        return SystemOneNoul(type="noul", instructions=question.instructions)
    if isinstance(question, Choice):
        return SystemOneChoice(
            type="choice",
            instructions=question.instructions,
            criteria={str(key): value for key, value in question.choices.items()},
        )
    if isinstance(question, Score):
        criteria: list[Any] = []
        for level in question.levels:
            if isinstance(level, str):
                criteria.append(level)
            elif level.description:
                criteria.append(f"{level.label}: {level.description}")
            else:
                criteria.append(level.label)
        return SystemOneScore(
            type="score",
            instructions=question.instructions,
            criteria=criteria,
        )
    raise TypeError(f"Unsupported decision question: {type(question).__name__}")


def _to_decision_response(response: Any) -> DecisionResponse:
    answers: dict[str, Any] = {}
    for name, answer in response.answers.items():
        if isinstance(answer, NoulAnswer):
            answers[name] = PredicateAnswer(probability=answer.noul)
        elif isinstance(answer, SystemOneChoiceAnswer):
            answers[name] = ChoiceAnswer(
                choice=answer.choice,
                probabilities=answer.probabilities,
                confidence=answer.confidence,
            )
        elif isinstance(answer, SystemOneScoreAnswer):
            answers[name] = ScoreAnswer(
                score=answer.score,
                legend={int(key): str(value) for key, value in answer.legend.items()},
                probabilities={int(key): value for key, value in answer.probabilities.items()},
                confidence=answer.confidence,
            )
    return DecisionResponse(
        model=response.model,
        answers=answers,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
        request_id=response.request_id,
    )


class BrainStormDecisions(RunnableSerializable[DecisionRequest, DecisionResponse]):
    """Ask typed decision questions through :class:`BrainStormS1`.

    The interface mirrors LangChain's decision model integrations. ``invoke``
    accepts ``{"input": <state>, "questions": {"name": <question>}}`` and
    returns a :class:`DecisionResponse`. It is a normal Runnable, so it can be
    composed in chains and used as a LangGraph node.

    Args:
        model: Decision model id. Defaults to ``typesafe/jev-1.14.0``.
        provider: ``"commandcode"``, ``"typesafe"``, ``"opencode"``, or a client
            supplied through ``client``.
        api_key: Overrides the provider's environment variable.
        base_url: Overrides the provider's default endpoint.
        timeout: Request timeout in seconds.
        max_retries: Retries for transient provider failures.
        zdr: Zero-data-retention header. Defaults to ``False`` because Command
            Code's ``/systemone`` refuses it.
        pricing: Cost lookup mode.
        client: Optional preconfigured ``BrainStormS1``.
        on_usage: Optional usage callback forwarded to the client.

    Example:
        >>> decisions = BrainStormDecisions()
        >>> result = decisions.invoke({
        ...     "input": "Production is down.",
        ...     "questions": {"urgent": Predicate(instructions="Is it urgent?")},
        ... })
        >>> result.predicates["urgent"].probability
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    model: str = "typesafe/jev-1.14.0"
    provider: str = "commandcode"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 2
    zdr: bool = False
    pricing: Literal["off", "cached", "any", "live"] = "any"
    client: Any = Field(default=None, exclude=True, repr=False)
    on_usage: Any = Field(default=None, exclude=True, repr=False)

    _client: BrainStormS1 = PrivateAttr()
    _owns_client: bool = PrivateAttr(default=True)

    def model_post_init(self, __context: Any) -> None:
        if self.client is not None:
            if not isinstance(self.client, BrainStormS1):
                raise TypeError("client must be a BrainStormS1 instance")
            self._client = self.client
            self._owns_client = False
            self.model = self._client.model
            self.provider = self._client.provider_name
            if self.on_usage is not None and self._client.on_usage is None:
                self._client.on_usage = self.on_usage
        else:
            self._client = BrainStormS1(
                model=self.model,
                provider=self.provider,
                api_key=self.api_key.get_secret_value() if self.api_key else None,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                zdr=self.zdr,
                pricing=self.pricing,
                on_usage=self.on_usage,
            )

    def invoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request synchronously."""

        return self._call_with_config(
            self._decide,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    async def ainvoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request asynchronously."""

        return await self._acall_with_config(
            self._adecide,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        if not isinstance(request, Mapping) or "input" not in request or "questions" not in request:
            raise ValueError(
                "Decision input must contain both 'input' and 'questions'."
            )
        state, assets = _to_state(request["input"])
        questions = {
            name: _to_systemone_question(question)
            for name, question in request["questions"].items()
        }
        try:
            response = self._client.system_one(
                state,
                questions,
                assets=assets or None,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        result = _to_decision_response(response)
        self._record_usage(result)
        return result

    async def _adecide(self, request: DecisionRequest) -> DecisionResponse:
        if not isinstance(request, Mapping) or "input" not in request or "questions" not in request:
            raise ValueError(
                "Decision input must contain both 'input' and 'questions'."
            )
        state, assets = _to_state(request["input"])
        questions = {
            name: _to_systemone_question(question)
            for name, question in request["questions"].items()
        }
        try:
            response = await self._client.asystem_one(
                state,
                questions,
                assets=assets or None,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        result = _to_decision_response(response)
        self._record_usage(result)
        return result

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": self.provider,
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config

    def _record_usage(self, response: DecisionResponse) -> None:
        try:
            from langsmith.run_helpers import get_current_run_tree

            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.extra.setdefault("metadata", {})["usage_metadata"] = {
                    "input_tokens": response.usage.input_tokens or 0,
                    "output_tokens": response.usage.output_tokens or 0,
                    "total_tokens": (response.usage.input_tokens or 0)
                    + (response.usage.output_tokens or 0),
                }
        except Exception:  # tracing must never break a decision
            logger.debug("Could not attach decision usage metadata.", exc_info=True)

    def make_tool(
        self,
        questions: Mapping[str, Any],
        *,
        name: str = "system_one_decisions",
        description: str | None = None,
    ) -> BaseTool:
        """Return a LangChain tool with fixed questions and a ``state`` input."""

        def _run(state: str) -> str:
            response = self.invoke(
                cast("DecisionRequest", {"input": state, "questions": dict(questions)})
            )
            return orjson.dumps(response.model_dump(mode="json")).decode("utf-8")

        async def _arun(state: str) -> str:
            response = await self.ainvoke(
                cast("DecisionRequest", {"input": state, "questions": dict(questions)})
            )
            return orjson.dumps(response.model_dump(mode="json")).decode("utf-8")

        question_summary = ", ".join(
            f"{key} ({getattr(value, 'type', type(value).__name__)})"
            for key, value in questions.items()
        )
        return StructuredTool.from_function(
            func=_run,
            coroutine=_arun,
            name=name,
            description=description
            or f"Evaluate System One decision questions: {question_summary}.",
        )

    def close(self) -> None:
        """Close the underlying BrainStormS1 client when this instance owns it."""

        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        """Close the underlying BrainStormS1 async client when owned."""

        if self._owns_client:
            await self._client.aclose()


def as_tool(
    decisions: BrainStormDecisions,
    questions: Mapping[str, Any],
    *,
    name: str = "system_one_decisions",
    description: str | None = None,
) -> BaseTool:
    """Create a LangChain tool from a decisions Runnable and fixed questions."""

    return decisions.make_tool(questions, name=name, description=description)


__all__ = [
    "Answer",
    "BrainStormDecisions",
    "Choice",
    "ChoiceAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "Level",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "RefusalAnswer",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
    "as_tool",
]
