"""Register v1 feature routers here as their implementation tasks are completed."""

from fastapi import APIRouter

from src.backend.api.v1.conversations import router as conversation_router

router = APIRouter()
router.include_router(conversation_router)
