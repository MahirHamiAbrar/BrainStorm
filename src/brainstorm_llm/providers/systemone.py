from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import httpx

from ..schemas.systemone import Image, SystemOneRequest, SystemOneResponse


class SystemOneWireMixin:
    systemone_path = "/systemone"

    if TYPE_CHECKING:
        def post(
            self,
            path: str,
            payload: Mapping[str, Any],
            *,
            headers: Mapping[str, str] | None = None,
            stream: bool = False,
        ) -> httpx.Response: ...

        async def apost(
            self,
            path: str,
            payload: Mapping[str, Any],
            *,
            headers: Mapping[str, str] | None = None,
            stream: bool = False,
        ) -> httpx.Response: ...

        @staticmethod
        def json_body(response: httpx.Response) -> dict[str, Any]: ...

    def _systemone_payload(
        self,
        request: SystemOneRequest,
        assets: Mapping[str, Image] | None,
    ) -> dict[str, Any]:
        payload = request.to_wire()
        if assets:
            payload["assets"] = {key: image.to_asset() for key, image in assets.items()}
        return payload

    def _parse_systemone(self, response: httpx.Response, payload: dict[str, Any]) -> SystemOneResponse:
        result = SystemOneResponse.model_validate(payload)
        result.request_id = (
            response.headers.get("x-typesafe-request-id")
            or response.headers.get("x-request-id")
            or response.headers.get("x-trace-id")
        )
        result.raw_http_response = response
        return result

    def system_one(
        self,
        request: SystemOneRequest,
        *,
        assets: Mapping[str, Image] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SystemOneResponse:
        payload = self._systemone_payload(request, assets)
        response = self.post(self.systemone_path, payload, headers=headers)
        return self._parse_systemone(response, self.json_body(response))

    async def asystem_one(
        self,
        request: SystemOneRequest,
        *,
        assets: Mapping[str, Image] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SystemOneResponse:
        payload = self._systemone_payload(request, assets)
        response = await self.apost(self.systemone_path, payload, headers=headers)
        return self._parse_systemone(response, self.json_body(response))
