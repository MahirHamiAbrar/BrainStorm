from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator

from .base import BrainStormModel, StrictModel

StateContent = str | dict[str, Any] | list[Any]
InstructionContent = str | dict[str, Any] | list[Any]


class NoulCriteria(BrainStormModel):
    """Optional descriptions of what counts as a yes or no answer."""

    true: InstructionContent | None = None
    false: InstructionContent | None = None


class Noul(BrainStormModel):
    """A yes/no System One question; the answer is a probability from 0 to 1."""

    type: Literal["noul"] = "noul"
    instructions: InstructionContent
    criteria: NoulCriteria | None = None


class Choice(BrainStormModel):
    """A multiple-choice question.

    ``criteria`` maps each option name to an optional description. Option keys
    are preserved even when their descriptions are ``None``.
    """

    type: Literal["choice"] = "choice"
    instructions: InstructionContent
    criteria: dict[str, str | None] | None = None

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, value: dict[str, str | None] | None) -> dict[str, str | None] | None:
        if value is not None and not value:
            raise ValueError("choice criteria must contain at least one option")
        return value


class Score(BrainStormModel):
    """A rubric question answered with an expected score.

    ``criteria`` lists rubric levels from lowest to highest (2-10 entries).
    """

    type: Literal["score"] = "score"
    instructions: InstructionContent
    criteria: list[Any] | None = None

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, value: list[Any] | None) -> list[Any] | None:
        if value is not None and not 2 <= len(value) <= 10:
            raise ValueError("score criteria must contain between 2 and 10 levels")
        return value


SystemOneQuestion = Annotated[
    Union[Noul, Choice, Score],
    Field(discriminator="type"),
]


class SystemOneRequest(StrictModel):
    """A validated System One request: one state and up to 100 questions.

    ``state`` may be a string, object, or array. Question names are echoed as
    the keys of the response's ``answers`` mapping.
    """

    model: str
    state: StateContent
    questions: dict[str, SystemOneQuestion] = Field(min_length=1, max_length=100)

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the TypeSafe wire shape, preserving choice option keys."""

        data = self.model_dump(mode="json")
        for question in data["questions"].values():
            criteria = question.get("criteria")
            if question.get("type") == "noul":
                if not criteria:
                    question.pop("criteria", None)
                else:
                    criteria = {key: value for key, value in criteria.items() if value is not None}
                    if criteria:
                        question["criteria"] = criteria
                    else:
                        question.pop("criteria", None)
            elif question.get("type") == "choice":
                if criteria is None:
                    question.pop("criteria", None)
            elif question.get("type") == "score" and criteria is None:
                question.pop("criteria", None)
        return data


def _detect_image_format(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"GIF8"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "png"


class Image:
    """An image asset referenced by name in System One state or questions.

    Args:
        source: A file path, ``bytes``, or raw image content.
        format: Optional ``png``, ``jpg``, or ``jpeg`` override. When omitted,
            it is inferred from the file suffix or the bytes.

    Raises:
        ValueError: if the image format cannot be represented in a System One request.
    """

    def __init__(self, source: str | Path | bytes, format: str | None = None) -> None:
        if isinstance(source, (str, Path)):
            self._data = Path(source).read_bytes()
            suffix = Path(source).suffix.lower().lstrip(".")
            self.format = format or ("jpg" if suffix == "jpeg" else suffix) or _detect_image_format(self._data)
        else:
            self._data = bytes(source)
            self.format = format or _detect_image_format(self._data)
        if self.format not in {"png", "jpg", "jpeg"}:
            self.format = _detect_image_format(self._data)
        if self.format not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"unsupported image format: {self.format}")

    @property
    def data(self) -> bytes:
        """Raw image bytes."""

        return self._data

    def to_asset(self) -> dict[str, str]:
        """Return the JSON asset object accepted by the System One API."""

        return {
            "type": "image",
            "format": self.format,
            "data_base64": base64.b64encode(self._data).decode("ascii"),
        }


class NoulAnswer(BrainStormModel):
    """Probability that a yes/no question is true."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BrainStormModel):
    """Selected choice, its confidence, and the probability of every option."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BrainStormModel):
    """Expected rubric score with its legend and per-level probabilities."""

    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, Any] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)


class BoxCoordinates(BrainStormModel):
    """A box's edges as fractions of image height and width (0-1)."""

    top: float
    bottom: float
    left: float
    right: float


class BoxAnswer(BrainStormModel):
    """A located image region and its confidence."""

    type: Literal["box"] = "box"
    box: BoxCoordinates
    confidence: float | None = None


SystemOneAnswer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, BoxAnswer],
    Field(discriminator="type"),
]


class SystemOneUsage(BrainStormModel):
    """Token counts reported by a System One request."""

    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(BrainStormModel):
    """Answers from a System One request.

    ``answers`` is keyed by the question names supplied in the request. Use the
    typed views :attr:`nouls`, :attr:`choices`, :attr:`scores`, and :attr:`boxes`.
    """

    model: str
    answers: dict[str, SystemOneAnswer] = Field(default_factory=dict)
    usage: SystemOneUsage = Field(default_factory=SystemOneUsage)
    request_id: str | None = Field(default=None, exclude=True)
    raw_http_response: Any = Field(default=None, exclude=True, repr=False)

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Yes/no answers keyed by question name."""

        return {key: value for key, value in self.answers.items() if isinstance(value, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Multiple-choice answers keyed by question name."""

        return {key: value for key, value in self.answers.items() if isinstance(value, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Rubric answers keyed by question name."""

        return {key: value for key, value in self.answers.items() if isinstance(value, ScoreAnswer)}

    @property
    def boxes(self) -> dict[str, BoxAnswer]:
        """Located-box answers keyed by question name."""

        return {key: value for key, value in self.answers.items() if isinstance(value, BoxAnswer)}
