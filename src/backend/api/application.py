"""Mount the v1 API without importing the AI engine or connecting to a database."""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.backend.api.errors import ErrorResponse, install_error_handlers
from src.backend.api.request_context import RequestContextMiddleware
from src.backend.api.v1.router import router
from src.backend.api.v1.schemas import ChatSettings


def install_api_foundation(app: FastAPI) -> FastAPI:
    """Return the mounted v1 app; the root retains legacy routes and error bodies."""
    versioned_api = FastAPI(
        title="LocalMind API v1",
        version="1.0.0",
        description=(
            "API rework foundation. Feature endpoints are added in subsequent tasks. "
            "Shared ChatSettings is a contract, not a legacy inference capability."
        ),
        responses={
            422: {"model": ErrorResponse, "description": "Request validation failed"},
            500: {"model": ErrorResponse, "description": "Unexpected server error"},
        },
    )
    install_error_handlers(versioned_api)
    versioned_api.include_router(router)

    def generate_openapi() -> dict[str, Any]:
        if versioned_api.openapi_schema is None:
            schema = get_openapi(
                title=versioned_api.title,
                version=versioned_api.version,
                description=versioned_api.description,
                routes=versioned_api.routes,
                servers=versioned_api.servers,
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            # Publish shared contracts before feature routes begin referencing them.
            for schema_model in (ChatSettings, ErrorResponse):
                model_schema = schema_model.model_json_schema(
                    ref_template="#/components/schemas/{model}"
                )
                components.update(model_schema.pop("$defs", {}))
                components[schema_model.__name__] = model_schema
            versioned_api.openapi_schema = schema
        return versioned_api.openapi_schema

    versioned_api.openapi = generate_openapi
    app.add_middleware(RequestContextMiddleware)
    app.mount("/api/v1", versioned_api, name="api-v1")
    return versioned_api
