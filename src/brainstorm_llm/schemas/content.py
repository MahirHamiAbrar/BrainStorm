from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field

from .base import BrainStormModel


class CacheControlEphemeral(BrainStormModel):
    """Mark the end of a reusable prompt prefix for provider-side caching."""

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


class Base64ImageSource(BrainStormModel):
    """Base64-encoded image bytes and their media type."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class URLImageSource(BrainStormModel):
    """An image referenced by a remote URL."""

    type: Literal["url"] = "url"
    url: str


class FileImageSource(BrainStormModel):
    """An image already uploaded to the provider's files API."""

    type: Literal["file"] = "file"
    file_id: str


ImageSource = Annotated[
    Union[Base64ImageSource, URLImageSource, FileImageSource],
    Field(discriminator="type"),
]


class PlainTextSource(BrainStormModel):
    """Plain-text document content."""

    type: Literal["text"] = "text"
    media_type: Literal["text/plain"] = "text/plain"
    data: str


class Base64PDFSource(BrainStormModel):
    """Base64-encoded PDF document content."""

    type: Literal["base64"] = "base64"
    media_type: Literal["application/pdf"] = "application/pdf"
    data: str


class URLPDFSource(BrainStormModel):
    """A PDF document referenced by a remote URL."""

    type: Literal["url"] = "url"
    url: str


class FileDocumentSource(BrainStormModel):
    """A document already uploaded to the provider's files API."""

    type: Literal["file"] = "file"
    file_id: str


DocumentSource = Annotated[
    Union[PlainTextSource, Base64PDFSource, URLPDFSource, FileDocumentSource],
    Field(discriminator="type"),
]


class TextBlock(BrainStormModel):
    """A block of plain text in a message or response."""

    type: Literal["text"] = "text"
    text: str
    citations: list[dict[str, Any]] | None = None
    cache_control: CacheControlEphemeral | None = None


class ImageBlock(BrainStormModel):
    """An image attached to a user message."""

    type: Literal["image"] = "image"
    source: ImageSource
    cache_control: CacheControlEphemeral | None = None


class DocumentBlock(BrainStormModel):
    """A text or PDF document attached to a user message."""

    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None
    context: str | None = None
    citations: list[dict[str, Any]] | None = None
    cache_control: CacheControlEphemeral | None = None


class ToolUseBlock(BrainStormModel):
    """A model request to call a tool, with parsed JSON input."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    cache_control: CacheControlEphemeral | None = None


class ToolResultBlock(BrainStormModel):
    """The result of a tool call, sent back in a user message."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list["TextBlock | ImageBlock | DocumentBlock"] | None = None
    is_error: bool | None = None
    cache_control: CacheControlEphemeral | None = None


class ThinkingBlock(BrainStormModel):
    """Model reasoning exposed by providers that return thinking traces."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class RedactedThinkingBlock(BrainStormModel):
    """Opaque, encrypted thinking content that must be passed back unchanged."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class ServerToolUseBlock(BrainStormModel):
    """A server-side tool invocation reported by Anthropic."""

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class UnknownContentBlock(BrainStormModel):
    """Fallback block for content types added by a provider after this release."""

    type: str


ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        DocumentBlock,
        ToolUseBlock,
        ToolResultBlock,
        ThinkingBlock,
        RedactedThinkingBlock,
        ServerToolUseBlock,
        UnknownContentBlock,
    ],
    Field(union_mode="left_to_right"),
]

ContentBlockParam = ContentBlock


