"""Check v1 errors, request IDs and payload validation without database or model startup."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
import httpx
from starlette.responses import StreamingResponse

from src.backend.api.application import install_api_foundation
from src.backend.api.errors import ApiError, ErrorResponse
from src.backend.api.v1.schemas import ChatSettings
from src.backend.database.schemas import ChatRequest


class ApiFoundationTests(unittest.TestCase):
    """Exercise real HTTP middleware/handlers through small routes registered only in this test
    app.
    """

    def setUp(self) -> None:
        """Build a fresh root/v1 app and test-only routes; close the HTTP client after each case."""
        self.app = FastAPI()
        self.versioned_api = install_api_foundation(self.app)

        @self.versioned_api.post("/_test/settings", response_model=ChatSettings)
        async def validate_settings(settings: ChatSettings) -> ChatSettings:
            """Return validated ChatSettings so cases can inspect defaults and reject invalid
            requests.
            """
            return settings

        @self.versioned_api.get("/_test/context")
        async def read_context(request: Request) -> dict[str, str]:
            """Yield once, then return this request's ID to detect accidental sharing between
            requests.
            """
            await asyncio.sleep(0)
            return {"request_id": request.state.request_id}

        @self.versioned_api.get("/_test/domain-error")
        async def domain_error() -> None:
            """Raise a known public ApiError so the case can inspect its code and details."""
            raise ApiError(
                status_code=409,
                code="RUN_ACTIVE",
                message="This conversation already has an active run",
                details={"run_id": "existing-run"},
            )

        @self.versioned_api.get("/_test/http-error/{status_code}")
        async def http_error(status_code: int) -> None:
            """Raise the selected HTTP status with private detail and Retry-After to check
            redaction/headers.
            """
            raise HTTPException(
                status_code=status_code,
                detail="Internal database credentials must stay private",
                headers={"Retry-After": "5"},
            )

        @self.versioned_api.get("/_test/unexpected-error")
        async def unexpected_error() -> None:
            """Raise private exception text that must not appear in the generic 500 response."""
            raise RuntimeError("secret connection string")

        @self.versioned_api.get("/_test/stream")
        async def stream() -> StreamingResponse:
            """Return a two-chunk stream to verify request-ID handling does not buffer or change
            its body.
            """

            async def chunks():
                """Yield two text chunks with an async scheduling point between them."""
                yield "first\n"
                await asyncio.sleep(0)
                yield "second\n"

            return StreamingResponse(chunks(), media_type="text/plain")

        @self.app.post("/_test/legacy-chat")
        async def legacy_chat(payload: ChatRequest) -> dict:
            """Echo the old ChatRequest fields to check v1 installation does not replace legacy
            validation.
            """
            return payload.model_dump(mode="json")

        @self.app.get("/_test/legacy-error")
        async def legacy_error() -> None:
            """Raise a legacy HTTP error whose original detail response must stay unchanged."""
            raise HTTPException(status_code=404, detail="Legacy error detail")

        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def assert_error(self, response: httpx.Response, status_code: int, code: str) -> dict:
        """Assert status/code and matching UUIDs in response header/body; return the parsed error
        dictionary.
        """
        self.assertEqual(response.status_code, status_code, response.text)
        envelope = ErrorResponse.model_validate(response.json())
        self.assertEqual(envelope.error.code, code)
        self.assertEqual(envelope.error.request_id, response.headers["X-Request-ID"])
        UUID(envelope.error.request_id)
        return envelope.model_dump()["error"]

    def test_chat_settings_defaults_and_explicit_values(self) -> None:
        """Check omitted settings receive defaults and explicitly supplied valid values are
        retained.
        """
        response = self.client.post("/api/v1/_test/settings", json={"model_name": "local-model"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model_name": "local-model",
                "personality": "general",
                "reasoning_mode": "low",
                "temperature": 0.0,
                "tools_enabled": True,
                "ranking_group_size": 3,
            },
        )
        response = self.client.post(
            "/api/v1/_test/settings",
            json={
                "model_name": "local-model",
                "personality": "researcher",
                "reasoning_mode": "high",
                "temperature": 0.7,
                "tools_enabled": False,
                "ranking_group_size": 5,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ranking_group_size"], 5)

    def test_all_supported_modes_personalities_and_group_sizes(self) -> None:
        """Accept every supported reasoning/personality value and allowed ranking group size."""
        for mode in ("low", "medium", "high", "extra"):
            for personality in ("general", "coder", "researcher", "genz"):
                for group_size in range(2, 8):
                    with self.subTest(mode=mode, personality=personality, group_size=group_size):
                        settings = ChatSettings(
                            model_name="local-model",
                            reasoning_mode=mode,
                            personality=personality,
                            ranking_group_size=group_size,
                        )
                        self.assertEqual(settings.ranking_group_size, group_size)

    def test_invalid_settings_are_rejected_without_echoing_input(self) -> None:
        """Reject invalid settings and unknown fields without returning submitted private values."""
        invalid_fields = [
            {"model_name": ""},
            {"model_name": "   "},
            {"model_name": 123},
            {"model_name": "a" * 256},
            {"reasoning_mode": "extreme"},
            {"personality": "unknown"},
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"temperature": "0.7"},
            {"temperature": True},
            {"tools_enabled": "false"},
            {"ranking_group_size": 1},
            {"ranking_group_size": 8},
            {"ranking_group_size": 2.5},
            {"ranking_group_size": 3.0},
            {"ranking_group_size": "3"},
            {"ranking_group_size": True},
            {"ranking_group_size": None},
            {"unknown_setting": "private-input-not-for-response"},
        ]
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                response = self.client.post(
                    "/api/v1/_test/settings", json={"model_name": "local-model", **fields}
                )
                error = self.assert_error(response, 422, "VALIDATION_ERROR")
                self.assertTrue(error["details"]["violations"])
                self.assertNotIn("private-input-not-for-response", response.text)
                for violation in error["details"]["violations"]:
                    self.assertEqual(set(violation), {"location", "code", "message"})

    def test_missing_model_name_has_field_location(self) -> None:
        """Report the missing model_name field through the standard validation details."""
        response = self.client.post("/api/v1/_test/settings", json={})
        error = self.assert_error(response, 422, "VALIDATION_ERROR")
        self.assertEqual(error["details"]["violations"][0]["location"], ["body", "model_name"])

    def test_malformed_json_and_non_finite_temperature_use_error_envelope(self) -> None:
        """Return the same 422 format for malformed JSON and invalid nonfinite temperatures."""
        for body in (
            '{"model_name":',
            '{"model_name":"local-model","temperature":NaN}',
            '{"model_name":"local-model","temperature":Infinity}',
        ):
            with self.subTest(body=body):
                response = self.client.post(
                    "/api/v1/_test/settings",
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                self.assert_error(response, 422, "VALIDATION_ERROR")

    def test_not_found_and_method_not_allowed_use_v1_errors(self) -> None:
        """Use v1 JSON errors for unknown paths and wrong HTTP methods."""
        self.assert_error(self.client.get("/api/v1/missing"), 404, "NOT_FOUND")
        response = self.client.get("/api/v1/_test/settings")
        self.assert_error(response, 405, "METHOD_NOT_ALLOWED")
        self.assertEqual(response.headers["Allow"], "POST")

    def test_domain_error_preserves_public_details(self) -> None:
        """Return the intended public error code/details from ApiError."""
        response = self.client.get("/api/v1/_test/domain-error")
        error = self.assert_error(response, 409, "RUN_ACTIVE")
        self.assertEqual(error["details"], {"run_id": "existing-run"})

    def test_http_server_errors_hide_internal_detail_and_preserve_headers(self) -> None:
        """Hide private HTTP server-error details while preserving protocol response headers."""
        for status_code, code in ((500, "INTERNAL_SERVER_ERROR"), (503, "SERVICE_UNAVAILABLE")):
            with self.subTest(status_code=status_code):
                response = self.client.get(f"/api/v1/_test/http-error/{status_code}")
                self.assert_error(response, status_code, code)
                self.assertNotIn("credentials", response.text)
                self.assertEqual(response.headers["Retry-After"], "5")

    def test_unexpected_error_is_correlated_and_does_not_leak_exception(self) -> None:
        """Return a matching request ID and generic 500 text without leaking the original
        exception.
        """
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            response = self.client.get("/api/v1/_test/unexpected-error")
        self.assert_error(response, 500, "INTERNAL_SERVER_ERROR")
        self.assertNotIn("secret connection string", response.text)
        self.assertTrue(any(response.headers["X-Request-ID"] in entry for entry in captured.output))
        self.assertFalse(any("secret connection string" in entry for entry in captured.output))

    def test_request_id_is_generated_or_normalized(self) -> None:
        """Generate missing/invalid IDs and normalize accepted UUID request headers."""
        accepted_id = str(uuid4())
        response = self.client.get(
            "/api/v1/_test/context", headers={"X-Request-ID": accepted_id.upper()}
        )
        self.assertEqual(response.headers["X-Request-ID"], accepted_id)
        self.assertEqual(response.json()["request_id"], accepted_id)
        for invalid_id in ("not-a-uuid", "x" * 1000):
            with self.subTest(invalid_id_length=len(invalid_id)):
                response = self.client.get(
                    "/api/v1/_test/context", headers={"X-Request-ID": invalid_id}
                )
                UUID(response.headers["X-Request-ID"])
                self.assertNotEqual(response.headers["X-Request-ID"], invalid_id)

    def test_concurrent_requests_have_independent_context(self) -> None:
        """Give simultaneous requests different IDs without overwriting each other's state."""

        async def send_concurrent_requests() -> list[httpx.Response]:
            """Send concurrent in-process requests and return their HTTP responses for ID
            comparison.
            """
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(
                    *(client.get("/api/v1/_test/context") for _ in range(12))
                )

        responses = asyncio.run(send_concurrent_requests())
        request_ids = {response.headers["X-Request-ID"] for response in responses}
        self.assertEqual(len(request_ids), 12)
        for response in responses:
            self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_streaming_response_keeps_request_id_and_body(self) -> None:
        """Keep response chunks and the request-ID header intact through the middleware."""
        response = self.client.get("/api/v1/_test/stream")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "first\nsecond\n")
        UUID(response.headers["X-Request-ID"])

    def test_legacy_validation_and_http_error_formats_are_preserved(self) -> None:
        """Keep old routes' detail errors unchanged after mounting v1."""
        response = self.client.post("/_test/legacy-chat", json={})
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        self.assertNotIn("error", response.json())
        response = self.client.get("/_test/legacy-error")
        self.assertEqual(response.json(), {"detail": "Legacy error detail"})
        response = self.client.get("/api/v10/missing")
        self.assertEqual(response.json(), {"detail": "Not Found"})

    def test_openapi_documents_shared_settings_and_error_contracts(self) -> None:
        """Publish settings/error fields and working docs URLs in the v1 OpenAPI document."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        components = schema["components"]["schemas"]
        settings = components["ChatSettings"]
        self.assertEqual(settings["properties"]["ranking_group_size"]["default"], 3)
        self.assertEqual(settings["properties"]["ranking_group_size"]["minimum"], 2)
        self.assertEqual(settings["properties"]["ranking_group_size"]["maximum"], 7)
        self.assertIn("ErrorResponse", components)
        self.assertIn("ErrorDetail", components)
        validation_schema = schema["paths"]["/_test/settings"]["post"]["responses"]["422"]
        self.assertEqual(
            validation_schema["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/ErrorResponse",
        )
        docs = self.client.get("/api/v1/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("/api/v1/openapi.json", docs.text)

    def test_foundation_does_not_publish_unimplemented_feature_endpoints(self) -> None:
        """Document implemented chat/submission routes while keeping future run endpoints absent."""
        app = FastAPI()
        install_api_foundation(app)
        with TestClient(app) as client:
            schema = client.get("/api/v1/openapi.json").json()
            self.assertIn("/conversations", schema["paths"])
            self.assertIn("/conversations/{conversation_id}/messages", schema["paths"])
            self.assertIn("/conversations/{conversation_id}/submissions", schema["paths"])
            self.assertFalse(any("/runs" in path for path in schema["paths"]))
            self.assertEqual(client.get("/openapi.json").json()["paths"], {})


class ProductionWiringTests(unittest.TestCase):
    """Check root legacy routes and v1 routes coexist without entering production startup."""

    def test_existing_conversation_route_and_v1_are_mounted_together(self) -> None:
        # TestClient without a lifespan context does not start DB/AI/model downloads.
        """Call the old conversation route with its service mocked and verify v1 is also mounted."""
        from src.backend.app import app
        from src.backend.routes.dependencies import get_conversation_service

        conversation_service = SimpleNamespace(get_conversations=AsyncMock(return_value=[]))
        app.dependency_overrides[get_conversation_service] = lambda: conversation_service
        try:
            with self.assertLogs("uvicorn.error", level="INFO") as captured:
                client = TestClient(app, raise_server_exceptions=False)
                try:
                    legacy_response = client.get("/conversation")
                    versioned_response = client.get("/api/v1/missing")
                    preflight_response = client.options(
                        "/api/v1/missing",
                        headers={
                            "Origin": "http://localhost:5500",
                            "Access-Control-Request-Method": "GET",
                        },
                    )
                    cors_response = client.get(
                        "/api/v1/missing", headers={"Origin": "http://localhost:5500"}
                    )
                finally:
                    client.close()
            self.assertEqual(legacy_response.status_code, 200)
            self.assertEqual(legacy_response.json(), [])
            self.assertEqual(versioned_response.json()["error"]["code"], "NOT_FOUND")
            self.assertEqual(preflight_response.status_code, 200)
            self.assertIn("X-Request-ID", cors_response.headers["Access-Control-Expose-Headers"])
            self.assertTrue(
                any(legacy_response.headers["X-Request-ID"] in entry for entry in captured.output)
            )
            conversation_service.get_conversations.assert_awaited_once()
        finally:
            del app.dependency_overrides[get_conversation_service]


if __name__ == "__main__":
    unittest.main()
